#ifndef __CONV_GRAPH_H__
#define __CONV_GRAPH_H__

#include <adf.h>
#include "conv.h"
#include "graph_concat.h"


/**
 * @defgroup Conv2D
 * 
 * @brief 2D convolution on H, W dimensions of BCHW/BHWC using kernels MCKK. Each
 * c-th KxK kernel is applied on C dimension. This is done over M iterations to yield
 * MxHxW per instance. This is done over B iterations to yield B batches.
 * 
 * @details
 * - std::conditional for kernel/graph typedef results in error in graph hierarchy algorithm
 * 
 * @tparam CONV     Conv2D Kernel
 * @tparam CONCAT   Concat Kernel (if multiinstance)
 * @tparam IS_BCHW  if BCHW or BHWC, affects concatenation (if multiinstance)
 * @tparam IS_KPAD  if kernel weights are padded, affects chunking (if multiinstance)
 * @tparam MCHUNK   M chunk size (if multiinstance)
 * @tparam INP_W    input width/height
 * @tparam OUT_W    output width/height, = INP_W - K/2
 * @tparam B        batch size
 * @tparam C        input channels
 * @tparam M        output channels
 * @tparam K        kernel width
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * Max size = 16384 and 4096 bytes respectively
 * 
 * @connections
 * @connect{pin[0], B*C*INP_W*INP_W*4}
 * @connect{pout[0], B*M*OUT_W*OUT_W*4}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K, int IS_RELU>
class ConvReluGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ConvReluGraph(
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      k[0] = adf::kernel::create_object<CONV<INP_W, OUT_W, B, C, M, K, IS_RELU>>(weights, bias);
      adf::source(k[0]) = "conv.cc";
      adf::headers(k[0]) = {"conv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (k[0].out[0], pout[0]);

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      // weights can be padded, not necessarily MCKK
      // separate bank not required for weights vs bias
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset((weights.size()*4+31)/32*32); // separate bank
    }

};


/**
 * @brief Single instance graph that streams weights and biases, significantly slower.
 * 
 * @connections
 * @connect{pin[0], B*C*INP_W*INP_W*4}
 * @connect{pin[1], M*K*K*C*4}
 * @connect{pin[2], M*4}
 * @connect{pout[0], B*M*OUT_W*OUT_W*4}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K, int IS_RELU>
class ConvReluGmemParamGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[3];
    adf::port<output> pout[1];

    ConvReluGmemParamGraph() { 
      k[0] = adf::kernel::create_object<CONV<INP_W, OUT_W, B, C, M, K, IS_RELU>>();
      adf::source(k[0]) = "conv.cc";
      adf::headers(k[0]) = {"conv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<M*K*K*C*4>>         (pin[1], k[0].in[1]);
      adf::connect<adf::window<M*4>>               (pin[2], k[0].in[2]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (k[0].out[0], pout[0]);
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases. 
 * If IS_BCHW=0 (using BHWC kernel): MCHUNK%8=0 and M%4=0. 
 * If IS_BCHW=1 (using BCHW kernel): MCHUNK*OUT_W*OUT_W%8=0 and M*OUT_W*OUT_W%4=0. 
 * Chunks MCKK weights by M dimension into MCHUNK chunks.
 * Each instance has max size = 16384 and 4096 bytes respectively.
 * Places maximum of 3x3 tiles, 8 conv tiles surrounding concat tile (max AIE DMA input=8)
 * 
 * @connections
 * @connect{pin[0:CHUNK_COUNT], B*C*INP_W*INP_W*4}
 * @connect{pout[0], B*M*OUT_W*OUT_W*4}
 * @endconnections
 */
template <
  template<int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int IS_BCHW, int IS_KPAD,
  int MCHUNK, int INP_W, int OUT_W, int B, int C, int M, int K, int IS_RELU>
class ConvReluChunkGraph : public adf::graph {

  private:
    static const int K2 = (IS_KPAD) ? (K+7)/8*8 : K;
    static const int MCUTCHUNK = M % MCHUNK;
    static const int CONCAT_CHUNK = (IS_BCHW) ? MCHUNK*OUT_W*OUT_W : MCHUNK;
    static const int CONCAT_BLOCK = (IS_BCHW) ? M*OUT_W*OUT_W : M;
    static const int CONCAT_CHUNK_COUNT = B*MCHUNK*OUT_W*OUT_W / CONCAT_CHUNK;

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
    static const int CHUNK_COUNT = (M + MCHUNK - 1) / MCHUNK; // ceiling
    adf::kernel convs[CHUNK_COUNT];
    ConcatGraph<CONCAT, float_t, CHUNK_COUNT, CONCAT_CHUNK_COUNT, CONCAT_CHUNK, CONCAT_BLOCK> concat_g;

    adf::port<input> pin[CHUNK_COUNT];
    adf::port<output> pout[1];

    ConvReluChunkGraph(
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      static_assert(CHUNK_COUNT <= 8);
      std::vector<float> wChunk;
      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*MCHUNK + MCHUNK > M) ? MCUTCHUNK : MCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*MCHUNK*K*K2*C, 
                                    weights.begin()+(i*MCHUNK+chunkSize)*K*K2*C); 
        wChunk.resize(MCHUNK*K*K2*C, 0);
        bChunk = std::vector<float>(bias.begin()+i*MCHUNK, bias.begin()+i*MCHUNK+chunkSize);
        bChunk.resize(MCHUNK, 0);

        convs[i] = adf::kernel::create_object<CONV<INP_W, OUT_W, B, C, MCHUNK, K, IS_RELU>>(wChunk, bChunk);
        adf::source(convs[i]) = "conv.cc";
        adf::headers(convs[i]) = {"conv.h"};
        adf::runtime<ratio>(convs[i]) = 0.6;
        
        adf::location<adf::kernel>(convs[i]) = adf::location<adf::kernel>(concat_g.k[0]) + 
          adf::relative_offset(tileOffsets[i]);
        adf::location_constraint tilePos = adf::location<adf::kernel>(convs[i]);
        adf::location<adf::parameter>(convs[i].param[0]) = tilePos;
        adf::location<adf::parameter>(convs[i].param[0]) = adf::offset(0);
        adf::location<adf::parameter>(convs[i].param[1]) = tilePos;
        adf::location<adf::parameter>(convs[i].param[1]) = adf::offset((MCHUNK*K*K2*C*4+31)/32*32);
        // input window and output window can be much larger
      }

      for (int i = 0; i < CHUNK_COUNT; i++) {
        adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[i], convs[i].in[0]);
        adf::connect<adf::window<B*OUT_W*OUT_W*MCHUNK*4>> (convs[i].out[0], concat_g.pin[i]);
      }
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (concat_g.pout[0], pout[0]);
    }
};
/** @} */


#endif // __CONV_GRAPH_H__