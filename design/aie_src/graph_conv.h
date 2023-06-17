#ifndef __CONV_GRAPH_H__
#define __CONV_GRAPH_H__

#include <adf.h>
#include "conv.h"
#include "pad.h"
#include "graph_concat.h"
#include "graph_split.h"


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
 * @tparam MCHUNK   M chunk size (if multiinstance)
 * @tparam HCHUNK   H chunk size (if multiinstance)
 * @tparam INP_W    input width/height
 * @tparam OUT_W    output width/height, = INP_W - K/2
 * @tparam STEP_H   stride H
 * @tparam STEP_W   stride W
 * @tparam B        batch size
 * @tparam C        input channels
 * @tparam M        output channels
 * @tparam K        kernel width
 * @tparam H0       Pixels added before height (default 0)
 * @tparam H1       Pixels added after height (default 0)
 * @tparam W0       Pixels added before width (default 0)
 * @tparam W1       Pixels added after width (default 0)
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * 
 * @connections
 * @connect{pin[0], B*C*INP_H*INP_W*4}
 * @connect{pout[0], B*M*OUT_H*OUT_W*4}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int, int, int, int> class CONV, 
  int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
  int B, int C, int M, int K, int IS_RELU, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    static constexpr int OUT_H = (PAD_H - K) / STEP_H + 1;

    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ConvReluGraph(
      std::vector<float> weights,
      std::vector<float> bias,
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<CONV<PAD_H, PAD_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU>>(weights, bias);
      adf::source(k[0]) = "conv.cc";
      adf::headers(k[0]) = {"conv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DScalar<float_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;
        adf::repetition_count(pad[0]) = repeat_cnt;

        adf::connect<adf::window<B*C*PAD_H*PAD_W*4>, adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W*4>> (pad[0].out[0], k[0].in[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W*4>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::window<B*M*OUT_H*OUT_W*4>> (k[0].out[0], pout[0]);

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
 * @connect{pin[0], B*C*INP_H*INP_W*4}
 * @connect{pin[1], stream}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W*4}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int, int, int, int> class CONV, 
  int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
  int B, int C, int M, int K, int IS_RELU, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    static constexpr int OUT_H = (PAD_H - K) / STEP_H + 1;

    adf::port<input> pin[2];
    adf::port<output> pout[1];

    ConvReluStreamGraph(
      std::vector<float> bias,
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<CONV<PAD_H, PAD_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU>>(bias);
      adf::source(k[0]) = "conv.cc";
      adf::headers(k[0]) = {"conv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      adf::heap_size(k[0]) = OUT_H*OUT_W*4 + 1024; // caches HoWo partial products

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DScalar<float_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;
        adf::repetition_count(pad[0]) = repeat_cnt;

        adf::connect<adf::window<B*C*PAD_H*PAD_W*4>, adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W*4>> (pad[0].out[0], k[0].in[0]);

        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W*4>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::stream> (pin[1], k[0].in[1]); // variable samples per iteration based on kernel
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      
      adf::samples_per_iteration(k[0].out[0]) = B*M*OUT_H*OUT_W;
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks MCKK weights by M dimension, maximum 8 chunks
 * If IS_BCHW=0 (using BHWC kernel): MCHUNK%8=0 and M%4=0. 
 * If IS_BCHW=1 (using BCHW kernel): MCHUNK*OUT_W*OUT_W%8=0 and M*OUT_W*OUT_W%4=0. 
 * 
 * @connections
 * @connect{pin[0], B*C*INP_W*INP_W*4}
 * @connect{pout[0], B*M*OUT_W*OUT_W*4}
 * @endconnections
 */
template <
  template<int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int IS_BCHW, int MCHUNK, 
  int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W,
  int B, int C, int M, int K, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkMGraph : public adf::graph {

  private:
    static constexpr int CHUNK_COUNT = (M + MCHUNK - 1) / MCHUNK; // ceiling
    static constexpr int CHUNK_REM = M % MCHUNK;

    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

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

    adf::kernel k[CHUNK_COUNT];
    std::vector<adf::kernel> pad;

  public:
    static constexpr int OUT_H = (PAD_H - K) / STEP_H + 1;
    static constexpr int CONCAT_W = (IS_BCHW) ? MCHUNK*OUT_H*OUT_W : MCHUNK;
    static constexpr int CONCAT_BLOCK = (IS_BCHW) ? M*OUT_H*OUT_W : M;
    static constexpr int CONCAT_H = (IS_BCHW) ? B : B*OUT_H*OUT_W;
    ConcatGraph<CONCAT, float_t, CHUNK_COUNT, CONCAT_H, CONCAT_W, CONCAT_BLOCK> concat_g;

    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ConvReluChunkMGraph(
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      static_assert(CHUNK_COUNT <= 8);
      std::vector<float> wChunk;
      std::vector<float> bChunk;
      int CKK = weights.size() / M;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*MCHUNK + MCHUNK > M) ? CHUNK_REM : MCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*MCHUNK*CKK, 
                                    weights.begin()+(i*MCHUNK+chunkSize)*CKK); 
        wChunk.resize(MCHUNK*CKK, 0);
        bChunk = std::vector<float>(bias.begin()+i*MCHUNK, bias.begin()+i*MCHUNK+chunkSize);
        bChunk.resize(MCHUNK, 0);

        k[i] = adf::kernel::create_object<CONV<PAD_H, PAD_W, OUT_W, STEP_H, STEP_W, B, C, MCHUNK, K, IS_RELU>>(wChunk, bChunk);
        adf::source(k[i]) = "conv.cc";
        adf::headers(k[i]) = {"conv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        
        adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(concat_g.k[0]) + 
          adf::relative_offset(tileOffsets[i]);
        adf::location_constraint tilePos = adf::location<adf::kernel>(k[i]);
        adf::location<adf::parameter>(k[i].param[0]) = tilePos;
        adf::location<adf::parameter>(k[i].param[0]) = adf::offset(0);
        adf::location<adf::parameter>(k[i].param[1]) = tilePos;
        adf::location<adf::parameter>(k[i].param[1]) = adf::offset((MCHUNK*CKK*4+31)/32*32);
        // input window and output window can be much larger
      }

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DScalar<float_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::window<B*C*INP_H*INP_W*4>, adf::stream> (pin[0], pad[0].in[0]);
        for (int i = 0; i < CHUNK_COUNT; i++)
          adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W*4>> (pad[0].out[0], k[i].in[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        for (int i = 0; i < CHUNK_COUNT; i++)
          adf::connect<adf::window<B*C*INP_H*INP_W*4>> (pin[0], k[i].in[0]);
      }

      for (int i = 0; i < CHUNK_COUNT; i++)
        adf::connect<adf::window<B*MCHUNK*OUT_H*OUT_W*4>> (k[i].out[0], concat_g.pin[i]);
      adf::connect<adf::stream, adf::window<B*M*OUT_H*OUT_W*4>> (concat_g.pout[0], pout[0]);
    }
};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by W dimension, maximum 8 chunks
 * If IS_BCHW=0 (using BHWC kernel): MCHUNK%8=0 and M%4=0. 
 * If IS_BCHW=1 (using BCHW kernel): MCHUNK*OUT_W*OUT_W%8=0 and M*OUT_W*OUT_W%4=0. 
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_W*INP_W*4}
 * @connect{pout[0], stream B*M*OUT_W*OUT_W*4}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W,
  int B, int C, int M, int K, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkHGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = K-STEP_H;
    typedef SplitGraph<SPLIT, float_t, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W> mSplitGraph;
    mSplitGraph split_graph;
    static constexpr int LCNT = mSplitGraph::LCNT;

    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - K) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - K) / STEP_H + 1;
    ConcatGraph<CONCAT, float_t, LCNT, B*M, HCHUNK_OUT*OUT_W, OUT_H*OUT_W> concat_graph;
    
  public:
    adf::port<adf::input> pin[2];
    adf::port<adf::output> pout[1];

    ConvReluChunkHGraph(
      std::vector<float> bias
    ) {
      static_assert((HCHUNK % STEP_H) == (K % STEP_H));

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DScalar<float_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream> (pad[0].out[0], split_graph.pin[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        adf::connect<adf::stream> (pin[0], split_graph.pin[0]);
      }

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<CONV<HCHUNK, PAD_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU>>(bias);
        adf::source(k[i]) = "conv.cc";
        adf::headers(k[i]) = {"conv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;

        adf::connect<adf::window<B*C*HCHUNK*PAD_W*4>> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::stream>                     (pin[1], k[i].in[1]);
        adf::connect<adf::stream, adf::window<B*M*HCHUNK_OUT*OUT_W*4>> (k[i].out[0], concat_graph.pin[i]);

        adf::samples_per_iteration(k[i].out[0]) = B*M*HCHUNK_OUT*OUT_W;

        adf::location<adf::parameter>(k[i].param[0]) = adf::location<adf::kernel>(k[i]);
        adf::location<adf::parameter>(k[i].param[0]) = adf::offset(0);
      }
      adf::connect<adf::stream> out_stream (concat_graph.pout[0], pout[0]);      
    }
};
/** @} */


#endif // __CONV_GRAPH_H__