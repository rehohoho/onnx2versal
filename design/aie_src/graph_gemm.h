#ifndef __GEMM_GRAPH_H_
#define __GEMM_GRAPH_H_

#include <assert.h>
#include <adf.h>
#include "gemm.h"
#include "graph_concat.h"
#include "graph_utils.h"


/**
 * @defgroup Gemm
 * 
 * @brief xA^T + b as per torch.nn.Linear. Applies general matrix multiply:
 * output(MxN) = input(MxK) * weights(KxN) + bias(N)
 * 
 * @details
 * - std::conditional for kernel/graph typedef results in error in graph hierarchy algorithm
 * 
 * @tparam GEMM     Gemm Kernel
 * @tparam CONCAT   Concat Kernel (if multiinstance)
 * @tparam NCHUNK   chunk size for N (if multiinstance)
 * @tparam M        number of rows of input matrix
 * @tparam K        number of cols / number of rows of weight matrix
 * @tparam N        number of cols of weight matrix / size of bias vector
 * @tparam NPAD     number of cols of weight matrix / size of bias vector, padded to vector 
 *                  boundary, used if weights are KxN
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * Max size = 16384 and 4096 bytes respectively
 * 
 * @connections
 * @connect{pin[0], M*K*4}
 * @connect{pout[0], M*N*4}
 * @endconnections
 */
template <template<int, int, int, int> class GEMM, 
  int M, int K, int N, int IS_RELU>
class GemmReluGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    GemmReluGraph(
      std::vector<float> weights,
      std::vector<float> bias,
      int repeat_cnt = 1
    ) { 
      static_assert(M*K*4 <= TILE_BYTES);
      static_assert(K*N*4 <= MAX_PARAM_BYTES);
      static_assert(M*N*4 <= TILE_BYTES);
      k[0] = adf::kernel::create_object<GEMM<M, K, N, IS_RELU>>(weights, bias);
      adf::source(k[0]) = "gemm.cc";
      adf::headers(k[0]) = {"gemm.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;

      adf::connect<adf::window<M*K*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<M*N*4>> (k[0].out[0], pout[0]);
      
      int offset = 0;
      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(offset);
      offset += (N*K*4 + 31)/32*32;
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset(offset); 
      offset += (N*4 + 31)/32*32;
      adf::location<adf::buffer>(k[0].in[0]) = tilePos;
      adf::location<adf::buffer>(k[0].in[0]) = {
        adf::offset(offset), 
        adf::offset(offset + (M*K*4+31)/32*32)};
    }

};


/**
 * @brief Single instance graph that streams weights and biases, significantly slower.
 * 
 * @connections
 * @connect{pin[0], M*K*4}
 * @connect{pin[1], K*N*4}
 * @connect{pin[2], N*4}
 * @connect{pout[0], M*N*4}
 * @endconnections
 */
template <template<int, int, int, int> class GEMM, 
  int M, int K, int N, int IS_RELU>
class GemmReluStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[2];
    adf::port<output> pout[1];

    GemmReluStreamGraph(
      std::vector<float> bias,
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<GEMM<M, K, N, IS_RELU>>(bias);
      adf::source(k[0]) = "gemm.cc";
      adf::headers(k[0]) = {"gemm.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (pin[1], k[0].in[1]);
      adf::connect<adf::window<M*N*4>> (k[0].out[0], pout[0]);
    }

};


/**
 * @brief Multiinstance graph for MxK times NxK that stores weights and biases.
 * Requires NxK weight, NCHUNK%8=0 and N%4=0
 * Chunks NxK weights by N dimension into NCHUNK chunks.
 * Each instance has max size = 16384 and 4096 bytes respectively.
 * Places maximum of 3x3 tiles, 8 conv tiles surrounding concat tile (max AIE DMA input=8)
 * Padding handled within graph, NPAD or KPAD parameters not used
 * 
 * @connections
 * @connect{pin[0:CHUNK_COUNT], M*K*4}
 * @connect{pout[0], M*N*4}
 * @endconnections
 */
template <
  template<int, int, int, int> class GEMM, 
  template<int, int, int, int> class CONCAT, 
  int NCHUNK, int M, int K, int N, int IS_RELU>
class GemmReluMknkChunkGraph : public adf::graph {

  private:
    static const int NCUTCHUNK = N % NCHUNK;

    adf::relative_coordinate tileOffsets[8] = {
      {.col_offset = -1, .row_offset = 0}, // left, right
      {.col_offset = 1, .row_offset = 0},
      {.col_offset = -1, .row_offset = 1}, // bottom row
      {.col_offset = 0, .row_offset = 1},
      {.col_offset = 1, .row_offset = 1},
      {.col_offset = -1, .row_offset = -1}, // top row
      {.col_offset = 0, .row_offset = -1},
      {.col_offset = 1, .row_offset = -1},
    };

  public:
    static const int CHUNK_COUNT = (N + NCHUNK - 1) / NCHUNK; // ceiling
    adf::kernel gemms[CHUNK_COUNT];
    ConcatGraph<CONCAT, CHUNK_COUNT, 1, NCHUNK, N> concat_g;
    
    adf::port<input> pin[CHUNK_COUNT];
    adf::port<output> pout[1];

    GemmReluMknkChunkGraph(
      std::vector<float> weights,
      std::vector<float> bias,
      int repeat_cnt = 1
    ) { 
      static_assert(M*K*4 <= TILE_BYTES);
      static_assert(K*N*4 <= MAX_PARAM_BYTES);
      static_assert(M*N*4 <= TILE_BYTES);
      
      std::vector<float> wChunk;
      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*NCHUNK + NCHUNK > N) ? NCUTCHUNK : NCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*NCHUNK*K, weights.begin()+(i*NCHUNK+chunkSize)*K); 
        wChunk.resize(NCHUNK*K, 0);
        bChunk = std::vector<float>(bias.begin()+i*NCHUNK, bias.begin()+i*NCHUNK+chunkSize);
        bChunk.resize(NCHUNK, 0);
        
        gemms[i] = adf::kernel::create_object<GEMM<M, K, NCHUNK, IS_RELU>>(wChunk, bChunk);
        adf::source(gemms[i]) = "gemm.cc";
        adf::headers(gemms[i]) = {"gemm.h"};
        adf::runtime<ratio>(gemms[i]) = 0.6;
        adf::repetition_count(gemms[i]) = repeat_cnt;

        adf::location<adf::kernel>(gemms[i]) = adf::location<adf::kernel>(concat_g.k[0]) + 
          adf::relative_offset(tileOffsets[i]);
        
        int offset = 0;
        adf::location_constraint tilePos = adf::location<adf::kernel>(gemms[i]);
        adf::location<adf::parameter>(gemms[i].param[0]) = tilePos; // weight (<= 16384B)
        adf::location<adf::parameter>(gemms[i].param[0]) = adf::offset(offset);
        offset += (NCHUNK*K*4 + 31)/32*32;
        adf::location<adf::parameter>(gemms[i].param[1]) = tilePos; // bias   (<= 4096B)
        adf::location<adf::parameter>(gemms[i].param[1]) = adf::offset(offset); 
        offset += (NCHUNK*4 + 31)/32*32;
        adf::location<adf::buffer>(gemms[i].in[0]) = tilePos;  // input window (<= 4096B)
        adf::location<adf::buffer>(gemms[i].in[0]) = {
          adf::offset(offset), 
          adf::offset(offset + (M*K*4+31)/32*32)};
      }

      for (int i = 0; i < CHUNK_COUNT; i++) {
        adf::connect<adf::window<M*K*4>> (pin[i], gemms[i].in[0]);
        adf::connect<adf::window<M*NCHUNK*4>> (gemms[i].out[0], concat_g.pin[i]);
      }

      adf::connect<adf::window<M*N*4>> (concat_g.pout[0], pout[0]);
    }

};


/**
 * @brief Multiinstance graph for MxK times KxN that stores weights and biases
 * Requires KxN_RND weight, NCHUNK%8=0, N%4=0
 * Chunks KxN weights by N dimension into NCHUNK chunks.
 * Each instance has max size = 16384 and 4096 bytes respectively.
 * Places maximum of 3x3 tiles, 8 conv tiles surrounding concat tile (max AIE DMA input=8)
 */
template <
  template<int, int, int, int> class GEMM, 
  template<int, int, int, int> class CONCAT, 
  int NCHUNK, int M, int K, int N, int IS_RELU>
class GemmReluMkknChunkGraph : public adf::graph {

  private:
    adf::relative_coordinate tileOffsets[8] = {
      {.col_offset = -1, .row_offset = 0}, // left, right
      {.col_offset = 1, .row_offset = 0},
      {.col_offset = -1, .row_offset = 1}, // bottom row
      {.col_offset = 0, .row_offset = 1},
      {.col_offset = 1, .row_offset = 1},
      {.col_offset = -1, .row_offset = -1}, // top row
      {.col_offset = 0, .row_offset = -1},
      {.col_offset = 1, .row_offset = -1},
    };

  public:
    static const int CHUNK_COUNT = (N + NCHUNK - 1) / NCHUNK; // ceiling
    adf::kernel gemms[CHUNK_COUNT];
    ConcatGraph<CONCAT, CHUNK_COUNT, M, NCHUNK, N> concat_g;
    
    adf::port<input> pin[CHUNK_COUNT];
    adf::port<output> pout[1];

    GemmReluMkknChunkGraph(
      std::vector<float> weights, // KxN_RND
      std::vector<float> bias,    // N
      int repeat_cnt = 1
    ) { 
      static_assert(M*K*4 <= TILE_BYTES);
      static_assert(K*NCHUNK*4 <= MAX_PARAM_BYTES);
      static_assert(M*NCHUNK*4 <= TILE_BYTES);

      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        
        // build wchunk
        std::vector<float> wChunk;
        wChunk.reserve(NCHUNK*K);
        for (int j = 0; j < K*N; j+=N) {
          wChunk.insert(wChunk.end(), weights.begin()+j+i*NCHUNK, weights.begin()+j+i*NCHUNK+NCHUNK);
        }
        
        // build bChunk
        bChunk = std::vector<float>(bias.begin()+i*NCHUNK, bias.begin()+i*NCHUNK+NCHUNK);
        bChunk.resize(NCHUNK, 0);
        
        gemms[i] = adf::kernel::create_object<GEMM<M, K, NCHUNK, IS_RELU>>(wChunk, bChunk);
        adf::source(gemms[i]) = "gemm.cc";
        adf::headers(gemms[i]) = {"gemm.h"};
        adf::runtime<ratio>(gemms[i]) = 0.6;
        adf::repetition_count(gemms[i]) = repeat_cnt;

        adf::location<adf::kernel>(gemms[i]) = adf::location<adf::kernel>(concat_g.k[0]) + 
          adf::relative_offset(tileOffsets[i]);
        int offset = 0;
        adf::location_constraint tilePos = adf::location<adf::kernel>(gemms[i]);
        adf::location<adf::parameter>(gemms[i].param[0]) = tilePos;
        adf::location<adf::parameter>(gemms[i].param[0]) = adf::offset(offset);
        offset += (NCHUNK*K*4 + 31)/32*32; // separate bank not required for weights vs bias
        adf::location<adf::parameter>(gemms[i].param[1]) = tilePos;
        adf::location<adf::parameter>(gemms[i].param[1]) = adf::offset(offset); 
        adf::location<adf::buffer>(gemms[i].in[0]) = tilePos;
        // adf::location<adf::buffer>(gemms[i].in[0]) = {
        //   adf::offset(offset), 
        //   adf::offset(offset + (M*K*4 + 4095)/4096*4096)};
      }

      for (int i = 0; i < CHUNK_COUNT; i++) {
        adf::connect<adf::window<M*K*4>> (pin[i], gemms[i].in[0]);
        adf::connect<adf::window<M*NCHUNK*4>> (gemms[i].out[0], concat_g.pin[i]);
      }
      adf::connect<adf::window<M*N*4>> (concat_g.pout[0], pout[0]);
    }

};
/** @} */


#endif // __GEMM_GRAPH_H_