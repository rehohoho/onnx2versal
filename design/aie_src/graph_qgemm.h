#ifndef __QGEMM_GRAPH_H_
#define __QGEMM_GRAPH_H_

#include <adf.h>
#include "qgemm.h"
#include "graph_concat.h"
#include "graph_utils.h"


/**
 * @defgroup Qgemm
 * 
 * @brief The qgemm operator consumes a quantized input tensor, its scale and zero point, 
 * a quantized weight, its scale and zero point, and output's scale and zero point, 
 * and computes the quantized output. xA^T + b as per torch.nn.Linear. 
 * Applies general matrix multiply: output(MxN) = input(MxK) * weights(KxN) + bias(N)
 * 
 * @tparam QGEMM    QGemm Kernel
 * @tparam TT       input/output dtype, int8_t or uint8_t
 * @tparam TTPARAM  weight dtype, int8_t or uint8_t
 * @tparam M        number of rows of input matrix
 * @tparam K        number of cols / number of rows of weight matrix
 * @tparam N        number of cols of weight matrix / size of bias vector
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * 
 * @connections
 * @connect{pin[0], M*K}
 * @connect{pout[0], stream M*N}
 * @endconnections
 */
template <template<typename, typename, int, int, int> class QGEMM, 
  typename TT, typename TTPARAM, int M, int K, int N>
class QgemmStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QgemmStreamGraph(
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero,
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<QGEMM<TT, TTPARAM, M, K, N>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qgemm.cc";
      adf::headers(k[0]) = {"qgemm.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset((K*N+31)/32*32);

      adf::connect<adf::window<M*K>> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      
      adf::samples_per_iteration(k[0].out[0]) = M*N;
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
  template<typename, typename, int, int, int> class QGEMM, 
  template<typename, int, int, int, int> class CONCAT, 
  int NCHUNK, 
  typename TT, typename TTPARAM, int M, int K, int N>
class QgemmChunkNStreamGraph : public adf::graph {

  private:
    adf::relative_coordinate tileOffsets[8] = {
      {.col_offset = -1, .row_offset = 1}, // top row
      {.col_offset = 0, .row_offset = 1},
      {.col_offset = 1, .row_offset = 1},
      {.col_offset = -1, .row_offset = -1}, // bottom row
      {.col_offset = 0, .row_offset = -1},
      {.col_offset = 1, .row_offset = -1},
      {.col_offset = -1, .row_offset = 0}, // left, right
      {.col_offset = 1, .row_offset = 0},
    };

  public:
    static const int CHUNK_COUNT = (N + NCHUNK - 1) / NCHUNK; // ceiling
    adf::kernel k[CHUNK_COUNT];
    ConcatGraph<CONCAT, TT, CHUNK_COUNT, M, NCHUNK, N> concat_g;
    
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QgemmChunkNStreamGraph(
      std::vector<TTPARAM> weights,  // KxN
      std::vector<int32_t> bias,    // N
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero,
      int repeat_cnt = 1
    ) { 
      static_assert(CHUNK_COUNT <= 8);
      static_assert(M*K <= TILE_BYTES);
      static_assert(K*NCHUNK <= MAX_PARAM_BYTES);
      static_assert(M*NCHUNK <= TILE_BYTES);

      std::vector<int32_t> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        
        // build wchunk
        std::vector<TTPARAM> wChunk;
        wChunk.reserve(NCHUNK*K);
        for (int j = 0; j < K*N; j+=N) {
          wChunk.insert(wChunk.end(), weights.begin()+j+i*NCHUNK, weights.begin()+j+i*NCHUNK+NCHUNK);
        }
        
        // build bChunk
        bChunk = std::vector<int32_t>(bias.begin()+i*NCHUNK, bias.begin()+i*NCHUNK+NCHUNK);
        bChunk.resize(NCHUNK, 0);
        
        k[i] = adf::kernel::create_object<QGEMM<TT, TTPARAM, M, K, NCHUNK>>(
          wChunk, bChunk, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
        adf::source(k[i]) = "qgemm.cc";
        adf::headers(k[i]) = {"qgemm.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::repetition_count(k[i]) = repeat_cnt;
        
        adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(concat_g.k[0]) + 
          adf::relative_offset(tileOffsets[i]);
        adf::location_constraint tilePos = adf::location<adf::kernel>(k[i]);
        adf::location<adf::parameter>(k[i].param[0]) = tilePos;
        adf::location<adf::parameter>(k[i].param[0]) = adf::offset(0);
        adf::location<adf::parameter>(k[i].param[1]) = tilePos;
        adf::location<adf::parameter>(k[i].param[1]) = adf::offset((K*NCHUNK+31)/32*32);
        // arbitrary input/output buffer location due to interconnect design
      }

      for (int i = 0; i < CHUNK_COUNT; i++) {
        adf::connect<adf::window<M*K>> (pin[0], k[i].in[0]);
        adf::connect<adf::stream> (k[i].out[0], concat_g.pin[i]);
        adf::samples_per_iteration(k[i].out[0]) = M*NCHUNK;
      }
      adf::connect<adf::stream> (concat_g.pout[0], pout[0]);
    }

};
/** @} */


#endif // __QGEMM_GRAPH_H_