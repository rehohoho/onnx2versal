#ifndef __CONV_GRAPH_H__
#define __CONV_GRAPH_H__

#include <type_traits>
#include <adf.h>
#include "concat.h"
#include "conv.h"
#include "pad.h"
#include "split.h"
#include "graph_concat.h"
#include "graph_split.h"
#include "graph_utils.h"


template <template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void set_heap_size(adf::kernel k) {
  if (
    (std::is_same<
     CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
     ConvReluScalarStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value)
  ) {
    adf::heap_size(k) = C/GROUP*KH*KW *4 + 1024; // caches CKK weights
  }
  else if (
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    ConvHx4ReluStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value) ||
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    ConvHx4Out4ReluStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value)
  ) {
    adf::heap_size(k) = C/GROUP*((KH*KW+3)/4*4) *4 + 1024; // caches CKK weights, padded to 4
  }
  else if (
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    ConvHx8ReluStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value)
  ) {
    adf::heap_size(k) = C/GROUP*KH*8 *4 + OUT_W_PAD*4 + 1024; // caches CKK weights, padded to 8 and one OUT_ROW
  }
  else if (
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    ConvHx4ReluStreamMultiRow<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value)
  ) {
    adf::heap_size(k) = C/GROUP*((KH*KW+3)/4*4) *4 + OUT_W_PAD*4 + 1024; // caches CKK weights and one OUT_ROW
  }
  else if (
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    Conv1x1ReluStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value) ||
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    Conv1x1Out4ReluStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value)
  ) {
    adf::heap_size(k) = (C/GROUP+3)/4*4 *4 + 1024; // caches CKK weights
  }
  else if ((std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    ConvHx4ReluPktStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value) ||
    (std::is_same<
    CONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>, 
    Conv1x1ReluPktStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>::value)
  ) {
    adf::heap_size(k) = 31712; // caches CKK weights, input window
  }
}

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
 * @tparam CONV       Conv2D Kernel
 * @tparam CONCAT     Concat Kernel (if multiinstance)
 * @tparam IS_BCHW    if BCHW or BHWC, affects concatenation (if multiinstance)
 * @tparam MCHUNK     M chunk size (if multiinstance)
 * @tparam HCHUNK     H chunk size (if multiinstance)
 * @tparam INP_H      input height
 * @tparam INP_W      input width
 * @tparam INP_W_PAD  input width
 * @tparam OUT_W      output width
 * @tparam OUT_W_PAD  output width padded to vector boundary
 * @tparam STEP_H     stride H
 * @tparam STEP_W     stride W
 * @tparam B          batch size
 * @tparam C          input channels
 * @tparam M          output channels
 * @tparam KH         kernel height
 * @tparam KW         kernel width
 * @tparam GROUP      split input into groups, C%GROUP==0
 * @tparam H0         Pixels added before height (default 0)
 * @tparam H1         Pixels added after height (default 0)
 * @tparam W0         Pixels added before width (default 0)
 * @tparam W1         Pixels added after width (default 0)
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * 
 * @connections
 * @connect{pin[0], B*C*INP_H*INP_W_PAD*4}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD*4}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;

    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ConvReluGraph(
      std::vector<float> weights,
      std::vector<float> bias,
      int repeat_cnt = 1
    ) { 
      static_assert(B*C*PAD_H*PAD_W*4 <= MAX_PARAM_BYTES);
      assert(weights.size()*4 <= MAX_PARAM_BYTES);
      static_assert(B*M*OUT_H*OUT_W_PAD*4 <= MAX_PARAM_BYTES);

      k[0] = adf::kernel::create_object<CONV<PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>(weights, bias);
      adf::source(k[0]) = "conv.cc";
      adf::headers(k[0]) = {"conv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamFloat<float_t, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;
        adf::repetition_count(pad[0]) = repeat_cnt;

        adf::connect<adf::window<B*C*INP_H*INP_W_PAD*4>, adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W*4>> (pad[0].out[0], k[0].in[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W_PAD*4>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::window<B*M*OUT_H*OUT_W_PAD*4>> (k[0].out[0], pout[0]);

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      // weights can be padded, not necessarily MCKK
      // separate bank not required for weights vs bias
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset((weights.size()*4+31)/32*32);
    }

};


/**
 * @brief Single instance graph that streams weights and biases, significantly slower.
 * 
 * @connections
 * @connect{pin[0], B*C*INP_H*INP_W_PAD*4}
 * @connect{pin[1], stream}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD*4}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;

    adf::port<input> pin[2];
    adf::port<output> pout[1];

    ConvReluStreamGraph(
      std::vector<float> bias
    ) { 
      static_assert(B*C*PAD_H*PAD_W*4 <= TILE_BYTES);
      
      k[0] = adf::kernel::create_object<CONV<PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>(bias);
      adf::source(k[0]) = "conv.cc";
      adf::headers(k[0]) = {"conv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::single_buffer(k[0].in[0]);

      set_heap_size<CONV,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>(k[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamFloat<float_t, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W*4>> (pad[0].out[0], k[0].in[0]);

        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      
      } else {
        adf::connect<adf::stream, adf::window<B*C*INP_H*INP_W_PAD*4>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::stream> (pin[1], k[0].in[1]); // variable samples per iteration based on kernel
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      
      adf::samples_per_iteration(k[0].out[0]) = B*M*OUT_H*OUT_W_PAD;
      
      adf::location<adf::buffer>(k[0].in[0]) = adf::location<adf::kernel>(k[0]);
      adf::location<adf::buffer>(k[0].in[0]) = adf::offset(0);
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks MCKK weights by M dimension, maximum 8 chunks
 * If IS_BCHW=0 (using BHWC kernel): MCHUNK%8=0 and M%4=0. 
 * If IS_BCHW=1 (using BCHW kernel): MCHUNK*OUT_W_PAD*OUT_W_PAD%8=0 and M*OUT_W_PAD*OUT_W_PAD%4=0. 
 * 
 * @connections
 * @connect{pin[0], B*C*INP_H*INP_W_PAD*4}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD*4}
 * @endconnections
 */
template <
  template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int IS_BCHW, int MCHUNK, 
  int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU,
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
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    static constexpr int CONCAT_W = (IS_BCHW) ? MCHUNK*OUT_H*OUT_W_PAD : MCHUNK;
    static constexpr int CONCAT_BLOCK = (IS_BCHW) ? M*OUT_H*OUT_W_PAD : M;
    static constexpr int CONCAT_H = (IS_BCHW) ? B : B*OUT_H*OUT_W_PAD;
    ConcatGraph<CONCAT, float_t, CHUNK_COUNT, CONCAT_H, CONCAT_W, CONCAT_BLOCK> concat_g;

    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ConvReluChunkMGraph(
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      static_assert(CHUNK_COUNT <= 8);
      static_assert(B*C*PAD_H*PAD_W*4 <= MAX_PARAM_BYTES);
      assert(weights.size() <= MAX_PARAM_BYTES*8);
      static_assert(B*M*OUT_H*OUT_W_PAD*4 <= MAX_PARAM_BYTES*8);

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

        k[i] = adf::kernel::create_object<CONV<PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,MCHUNK,KH,KW,GROUP,IS_RELU>>(wChunk, bChunk);
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
          adf::kernel::create_object<Pad2DStreamFloat<float_t, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::window<B*C*INP_H*INP_W_PAD*4>, adf::stream> (pin[0], pad[0].in[0]);
        for (int i = 0; i < CHUNK_COUNT; i++)
          adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W*4>> (pad[0].out[0], k[i].in[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        for (int i = 0; i < CHUNK_COUNT; i++)
          adf::connect<adf::window<B*C*INP_H*INP_W_PAD*4>> (pin[0], k[i].in[0]);
      }

      for (int i = 0; i < CHUNK_COUNT; i++)
        adf::connect<adf::window<B*MCHUNK*OUT_H*OUT_W_PAD*4>> (k[i].out[0], concat_g.pin[i]);
      adf::connect<adf::stream> (concat_g.pout[0], pout[0]);
    }
};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by H dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_H*INP_W_PAD*4}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD*4}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkHGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = KH-STEP_H;
    typedef SplitGraph<SPLIT, float_t, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W> mSplitGraph;
    mSplitGraph split_graph;
    static constexpr int LCNT = mSplitGraph::LCNT;

    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - KH) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    ConcatStreamGraph<CONCAT, float_t, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;

    adf::relative_coordinate tileOffsets[8] = {
      {.col_offset = -1, .row_offset = 1}, // top left, clockwise
      {.col_offset = 0, .row_offset = 2},
      {.col_offset = 0, .row_offset = 1},
      {.col_offset = 1, .row_offset = 0},
      {.col_offset = 0, .row_offset = -1},
      {.col_offset = 0, .row_offset = -2},
      {.col_offset = -1, .row_offset = -1},
      {.col_offset = -1, .row_offset = 0},
    };

    adf::relative_coordinate concat_k1_offsets[4] = {
      {.col_offset = -1, .row_offset = 2}, // top left, clockwise
      {.col_offset = 1, .row_offset = 1},
      {.col_offset = 1, .row_offset = -1},
      {.col_offset = -1, .row_offset = -2},
    };
    
  public:
    adf::port<adf::input> pin[2];
    adf::port<adf::output> pout[1];

    ConvReluChunkHGraph(
      std::vector<float> bias
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));
      static_assert(LCNT <= 8);
      static_assert(B*C*PAD_H*PAD_W*4 + B*C*(KH-1)*7*PAD_W*4 <= MAX_PARAM_BYTES*8);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamFloat<float_t, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream> (pad[0].out[0], split_graph.pin[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
        // split and pad can't be placed on same tile due to stream co-placement constraints
      } else {
        adf::connect<adf::stream> (pin[0], split_graph.pin[0]);
      }

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<CONV<HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>(bias);
        adf::source(k[i]) = "conv.cc";
        adf::headers(k[i]) = {"conv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::single_buffer(k[i].in[0]);

        set_heap_size<CONV,HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>(k[i]);

        adf::connect<adf::window<B*C*HCHUNK*PAD_W*4>> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::stream>                     (pin[1], k[i].in[1]);
        adf::connect<adf::stream>                     (k[i].out[0], concat_graph.pin[i]);

        adf::samples_per_iteration(k[i].out[0]) = B*M*HCHUNK_OUT*OUT_W_PAD;

        adf::location<adf::kernel>(k[i]) = 
          adf::location<adf::kernel>(split_graph.k[0]) + adf::relative_offset(tileOffsets[i]);
        adf::location_constraint tilePos = adf::location<adf::kernel>(k[i]);
        adf::location<adf::parameter>(k[i].param[0]) = tilePos; // may bust tiles adjacent to split
        adf::location<adf::parameter>(k[i].param[0]) = adf::offset(0);
      }
      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(split_graph.k[0]) + adf::relative_offset(concat_k1_offsets[i]);
      }
    }
};


/**
 * @brief Multiinstance graph that stores biases, 
 * chunks BCHW by H dimension
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_H*INP_W_PAD*4}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD*4}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkHStreamGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = KH-STEP_H;
    static constexpr int LCNT = (PAD_H - HCHUNK) / (HCHUNK - OVERLAP) + 1;
    adf::kernel split[(LCNT+1)/2];
    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - KH) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    ConcatStreamGraph<CONCAT, float_t, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;
    
  public:
    adf::port<adf::input> pin[2];
    adf::port<adf::output> pout[1];

    ConvReluChunkHStreamGraph(
      std::vector<float> bias
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));

      for (int i = 0; i < LCNT/2; i++) {
        split[i] = adf::kernel::create_object<SplitFilterFloatStreamTwice<float_t, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W>>(i*2);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::samples_per_iteration(split[i].in[0]) = B*C*PAD_H*PAD_W;
        adf::samples_per_iteration(split[i].out[0]) = B*C*HCHUNK*PAD_W;
        adf::samples_per_iteration(split[i].out[1]) = B*C*HCHUNK*PAD_W;
      }
      if ((LCNT & 0x1) == 1) {
        int i = (LCNT+1)/2 - 1;
        split[i] = adf::kernel::create_object<SplitFilterFloatStream<float_t, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W>>(LCNT-1);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::samples_per_iteration(split[i].in[0]) = B*C*PAD_H*PAD_W;
        adf::samples_per_iteration(split[i].out[0]) = B*C*HCHUNK*PAD_W;
      }
      
      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<CONV<HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>(bias);
        adf::source(k[i]) = "conv.cc";
        adf::headers(k[i]) = {"conv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::single_buffer(k[i].in[0]);

        set_heap_size<CONV,HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>(k[i]);

        adf::connect<adf::stream, adf::window<B*C*HCHUNK*PAD_W*4>> (split[i/2].out[i&0x1], k[i].in[0]);
        adf::connect<adf::stream>                     (pin[1], k[i].in[1]);
        adf::connect<adf::stream>                     (k[i].out[0], concat_graph.pin[i]);

        adf::samples_per_iteration(k[i].out[0]) = B*M*HCHUNK_OUT*OUT_W_PAD;

        adf::location<adf::buffer>(k[i].in[0]) = adf::location<adf::kernel>(k[i]);
        adf::location<adf::buffer>(k[i].in[0]) = {adf::offset(0)};
      }

      for (int i = 0; i < (LCNT+1)/2; i++) {
        adf::location<adf::kernel>(split[i]) = 
          adf::location<adf::kernel>(k[i*2]) + adf::relative_offset({.col_offset=1, .row_offset=0});
        
        adf::location_constraint sTilePos = adf::location<adf::kernel>(split[i]);
        adf::location<adf::stack>(split[i]) = sTilePos;
        adf::location<adf::stack>(k[i*2]) = sTilePos;
        adf::location<adf::parameter>(k[i*2].param[0]) = sTilePos;
        
        if (i*2+1 < LCNT) {
          adf::location<adf::kernel>(k[i*2+1]) = sTilePos + adf::relative_offset({.col_offset=0, .row_offset=1});
        }
        // if (i*2+2 < LCNT) {
        //   adf::location<adf::kernel>(k[i*2+2]) = sTilePos + adf::relative_offset({.col_offset=2, .row_offset=0});
        // }
      }
      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamFloat<float_t, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        for (int i = 0; i < (LCNT+1)/2; i++)
          adf::connect<adf::stream> (pad[0].out[0], split[i].in[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
        // split and pad can't be placed on same tile due to stream co-placement constraints
      } else {
        for (int i = 0; i < (LCNT+1)/2; i++)
          adf::connect<adf::stream> (pin[0], split[i].in[0]);
      }

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(k[i*2]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        
        adf::location_constraint cTilePos = adf::location<adf::kernel>(concat_graph.k1[i]);
        adf::location<adf::parameter>(k[i*2+1].param[0]) = cTilePos;
        adf::location<adf::stack>(k[i*2+1]) = cTilePos;
        adf::location<adf::stack>(concat_graph.k1[i]) = cTilePos;
      }
    }
};


/**
 * @brief Multiinstance graph that stores biases, 
 * chunks BCHW by H dimension
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_H*INP_W_PAD*4}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD*4}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkHPktStreamGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = KH-STEP_H;
    typedef SplitFilterPktStreamGraph<SPLIT, float_t, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W> mSplitGraph;
    mSplitGraph split_graph;

    static constexpr int LCNT = (PAD_H - HCHUNK) / (HCHUNK - OVERLAP) + 1;
    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - KH) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    ConcatStreamGraph<CONCAT, float_t, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;

  public:
    adf::port<adf::input> pin[2];
    adf::port<adf::output> pout[2];

    ConvReluChunkHPktStreamGraph(
      std::vector<float> bias
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));
      
      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<CONV<HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>>(bias);
        adf::source(k[i]) = "conv.cc";
        adf::headers(k[i]) = {"conv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;

        set_heap_size<CONV,HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP,IS_RELU>(k[i]);

        adf::connect<adf::pktstream> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::stream>    (pin[1], k[i].in[1]);
        adf::connect<adf::stream>    (k[i].out[0], concat_graph.pin[i]);

        adf::samples_per_iteration(k[i].out[0]) = B*M*HCHUNK_OUT*OUT_W_PAD;

        if ((i&0x1) == 1) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        }
        if (i == 2) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=0, .row_offset=2});
        }
        adf::location<adf::stack>(k[i]) = adf::location<adf::kernel>(k[i]);
        
      }

      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamFloat<float_t, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream> (pad[0].out[0], split_graph.pin[0]);
        adf::connect<adf::stream> (pad[0].out[0], pout[1]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
        // split and pad can't be placed on same tile due to stream co-placement constraints
      } else {
        adf::connect<adf::stream> (pin[0], split_graph.pin[0]);
      }

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(k[i*2+1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        
        adf::location_constraint cTilePos = adf::location<adf::kernel>(concat_graph.k1[i]);
        adf::location<adf::parameter>(k[i*2+1].param[0]) = cTilePos;
        adf::location<adf::stack>(concat_graph.k1[i]) = cTilePos;
      }
    }
};
/** @} */


#endif // __CONV_GRAPH_H__