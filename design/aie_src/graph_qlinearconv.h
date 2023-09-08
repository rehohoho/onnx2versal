#ifndef __QLINEARCONV_GRAPH_H__
#define __QLINEARCONV_GRAPH_H__

#include <adf.h>
#include "qlinearconv.h"
#include "pad.h"
#include "split.h"
#include "graph_concat.h"
#include "graph_split.h"
#include "graph_utils.h"


template <template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP>
void set_heap_size(adf::kernel k) {
  if (
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvScalarStream<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx4Stream<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx4StreamScale32bit<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx6x8bitStream<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value)
  ) {
    adf::heap_size(k) = C/GROUP*((KH*KW+15)/16*16) + 1024; // caches CKK weights
  }
  else if (
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx4PktStream<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) || 
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx4Stream_0<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx4Stream_1<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvHx4Stream_2<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConv1x1PktStream<TT,TTPARAM,INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value)
    
  ) {
    adf::heap_size(k) = 31712; // caches CKK weights, input window
  }
}


/**
 * @defgroup QLinearConv
 * 
 * @brief The convolution operator consumes a quantized input tensor, its scale and zero point, 
 * a quantized filter, its scale and zero point, and output's scale and zero point, 
 * and computes the quantized output. 2D convolution on H, W dimensions of BCHW using kernels MCKK. 
 * Each c-th KxK kernel is applied on C dimension. This is done over M iterations to yield
 * MxHxW per instance. This is done over B iterations to yield B batches.
 * 
 * @tparam QLINEARCONV  Conv2D Kernel
 * @tparam TT           input/output dtype, int8_t or uint8_t only
 * @tparam TTPARAM      weight dtype, int8_t or uint8_t only
 * @tparam INP_H        input height
 * @tparam INP_W        input width to use
 * @tparam INP_W_PAD    input width, padded to vector boundary
 * @tparam OUT_W        output width to use, unable to infer due to width padding
 * @tparam OUT_W_PAD    output width, padded to vector boundary
 * @tparam STEP_H       stride in height dimension
 * @tparam STEP_W       stride in width dimension
 * @tparam B            batch size
 * @tparam C            input channels
 * @tparam M            output channels
 * @tparam KH           kernel height
 * @tparam KW           kernel width
 * @tparam GROUP        split input into groups, C%GROUP==0
 * @tparam H0           Pixels added before height (default 0)
 * @tparam H1           Pixels added after height (default 0)
 * @tparam W0           Pixels added before width (default 0)
 * @tparam W1           Pixels added after width (default 0)
 *
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * Max size = 16384 and 4096 bytes respectively
 * 
 * @connections
 * @connect{pin[0], B*C*INP_H*INP_W_PAD}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, int, int, int, int, int, int, int, int> class PAD,
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;

    QLinearConvGraph(
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) { 
      static_assert(B*C*PAD_H*PAD_W <= MAX_PARAM_BYTES);
      assert(weights.size() <= MAX_PARAM_BYTES);
      static_assert(B*M*OUT_H*OUT_W_PAD <= MAX_PARAM_BYTES);
      
      k[0] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<PAD<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W>> (pad[0].out[0], k[0].in[0]);
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W_PAD>> (pin[0], k[0].in[0]);
      }

      adf::connect<adf::window<B*M*OUT_H*OUT_W_PAD>> (k[0].out[0], pout[0]);

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
    }

};


/**
 * @brief Single instance graph that streams weights and biases, significantly slower.
 * 
 * @connections
 * @connect{pin[0], B*C*INP_H*INP_W_PAD}
 * @connect{pin[1], stream}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, int, int, int, int, int, int, int, int> class PAD,
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;

    adf::vector<adf::port<input>> pin;
    adf::port<output> pout[1];

    void init_helper(TT x_zero) {
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      if (B*C*PAD_H*PAD_W > MAX_PARAM_BYTES)
        adf::single_buffer(k[0].in[0]);
      
      set_heap_size<QLINEARCONV,TT,TTPARAM,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>(k[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<PAD<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W>> (pad[0].out[0], k[0].in[0]);

        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W_PAD;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;

        adf::location<adf::kernel>(pad[0]) = adf::location<adf::kernel>(k[0]) + 
          adf::relative_offset({.col_offset=0, .row_offset=1});
        
        adf::location_constraint padTile = adf::location<adf::kernel>(pad[0]);
        adf::location<adf::stack>(pad[0]) = padTile;
        adf::location<adf::stack>(k[0]) = padTile;
        adf::location<adf::parameter>(k[0].param[0]) = padTile;
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W_PAD>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::samples_per_iteration(k[0].out[0]) = B*M*OUT_H*OUT_W_PAD;
      
      if (B*C*PAD_H*PAD_W > MAX_PARAM_BYTES) {
        adf::location<adf::buffer>(k[0].in[0]) = {adf::offset(0)};
      } else {
        adf::location<adf::buffer>(k[0].in[0]) = {adf::offset(0), adf::offset(16384)};
      }  
    }

    QLinearConvStreamGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) { 
      static_assert(B*C*PAD_H*PAD_W <= TILE_BYTES);
      k[0] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
        bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      
      adf::port<adf::input> pin0;
      adf::port<adf::input> pin1;
      pin.push_back(pin0);
      pin.push_back(pin1);

      adf::connect<adf::stream> (pin[1], k[0].in[1]); // variable samples per iteration based on kernel
      
      init_helper(x_zero);
    }

    QLinearConvStreamGraph(
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) { 
      static_assert(B*C*PAD_H*PAD_W <= MAX_PARAM_BYTES);
      assert(weights.size() <= MAX_PARAM_BYTES);
      static_assert(B*M*OUT_H*OUT_W_PAD <= MAX_PARAM_BYTES);

      k[0] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      
      adf::port<adf::input> pin0;
      pin.push_back(pin0);

      init_helper(x_zero);
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by H dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_H*INP_W_PAD}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = KH-STEP_H;
    typedef SplitGraph<SplitInt8, TT, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W> mSplitGraph;
    mSplitGraph split_graph;
    static constexpr int LCNT = mSplitGraph::LCNT;

    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - KH) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    ConcatStreamGraph<CONCAT, TT, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;

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

    QLinearConvChunkHGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));
      static_assert(LCNT <= 8);
      static_assert(B*C*HCHUNK*PAD_W <= TILE_BYTES);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamInt8<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
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
        k[i] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, HCHUNK, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
          bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
        adf::source(k[i]) = "qlinearconv.cc";
        adf::headers(k[i]) = {"qlinearconv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        if (B*C*HCHUNK*PAD_W > MAX_PARAM_BYTES)
          adf::single_buffer(k[i].in[0]);

        set_heap_size<QLINEARCONV,TT,TTPARAM,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>(k[i]);

        adf::connect<adf::window<B*C*HCHUNK*PAD_W>> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::stream>                   (pin[1], k[i].in[1]);
        adf::connect<adf::stream>                   (k[i].out[0], concat_graph.pin[i]);

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
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by H dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_W*INP_W_PAD}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHStreamGraph : public adf::graph {

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
    ConcatStreamGraph<CONCAT, TT, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;
    
  public:
    adf::port<adf::input> pin[2];
    adf::port<adf::output> pout[1];

    QLinearConvChunkHStreamGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) {
      static_assert((HCHUNK - KH + 1) % STEP_H == 0);
      static_assert(B*C*HCHUNK*PAD_W <= TILE_BYTES);

      for (int i = 0; i < LCNT/2; i++) {
        split[i] = adf::kernel::create_object<SplitFilterInt8StreamTwice<TT, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W>>(i*2);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::samples_per_iteration(split[i].in[0]) = B*C*PAD_H*PAD_W;
        adf::samples_per_iteration(split[i].out[0]) = B*C*HCHUNK*PAD_W;
        adf::samples_per_iteration(split[i].out[1]) = B*C*HCHUNK*PAD_W;
      }
      if ((LCNT & 0x1) == 1) {
        int i = (LCNT+1)/2 - 1;
        split[i] = adf::kernel::create_object<SplitFilterInt8Stream<TT, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W>>(LCNT-1);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::samples_per_iteration(split[i].in[0]) = B*C*PAD_H*PAD_W;
        adf::samples_per_iteration(split[i].out[0]) = B*C*HCHUNK*PAD_W;
      }

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, HCHUNK, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
          bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
        adf::source(k[i]) = "qlinearconv.cc";
        adf::headers(k[i]) = {"qlinearconv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        if (B*C*HCHUNK*PAD_W > MAX_PARAM_BYTES)
          adf::single_buffer(k[i].in[0]);

        set_heap_size<QLINEARCONV,TT,TTPARAM,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>(k[i]);

        adf::connect<adf::window<B*C*HCHUNK*PAD_W>> (split[i/2].out[i&0x1], k[i].in[0]);
        adf::connect<adf::stream>                   (pin[1], k[i].in[1]);
        adf::connect<adf::stream>                   (k[i].out[0], concat_graph.pin[i]);

        if ((i & 0x1) != 0) {
          adf::location<adf::kernel>(k[i]) =
            adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=1, .row_offset=0});
          adf::location<adf::kernel>(split[i/2]) = 
            adf::location<adf::kernel>(k[i]) + adf::relative_offset({.col_offset=0, .row_offset=-1});
          
          adf::location_constraint sTilePos = adf::location<adf::kernel>(split[i/2]);
          adf::location<adf::stack>(split[i/2]) = sTilePos;
          adf::location<adf::stack>(k[i]) = sTilePos;
          adf::location<adf::parameter>(k[i].param[0]) = sTilePos;
          adf::location<adf::parameter>(k[i].param[0]) = adf::offset(0);
        }
        
        adf::location_constraint kTilePos = adf::location<adf::kernel>(k[i]);
        adf::location<adf::buffer>(k[i].in[0]) = kTilePos; // may bust tiles adjacent to split
        if (B*C*HCHUNK*PAD_W > MAX_PARAM_BYTES) {
          adf::location<adf::buffer>(k[i].in[0]) = {adf::offset(0)};
        } else {
          adf::location<adf::buffer>(k[i].in[0]) = {adf::offset(0), adf::offset(16384)};
        }        
      }
      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamInt8<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
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
        adf::location<adf::parameter>(k[i*2].param[0]) = cTilePos;
        adf::location<adf::parameter>(k[i*2].param[0]) = adf::offset(0);
        adf::location<adf::stack>(k[i*2]) = cTilePos;
        adf::location<adf::stack>(concat_graph.k1[i]) = cTilePos;
      }
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by H dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_W*INP_W_PAD}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHPktStreamGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = KH-STEP_H;
    typedef SplitFilterPktStreamGraph<SplitFilterInt8PktStream, TT, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W> mSplitGraph;
    mSplitGraph split_graph;

    static constexpr int LCNT = (PAD_H - HCHUNK) / (HCHUNK - OVERLAP) + 1;
    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - KH) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    ConcatStreamGraph<CONCAT, TT, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;
    
  public:
    std::vector<adf::port<adf::input>> pin;
    adf::port<adf::output> pout[1];

    void init_helper(TT x_zero) {
      for (int i = 0; i < LCNT; i++) {
        adf::source(k[i]) = "qlinearconv.cc";
        adf::headers(k[i]) = {"qlinearconv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;

        set_heap_size<QLINEARCONV,TT,TTPARAM,HCHUNK,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>(k[i]);

        adf::connect<adf::pktstream> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::stream> (k[i].out[0], concat_graph.pin[i]);
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
          adf::kernel::create_object<Pad2DStreamInt8<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
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

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(k[i*2+1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        
        adf::location_constraint cTilePos = adf::location<adf::kernel>(concat_graph.k1[i]);
        adf::location<adf::parameter>(k[i*2+1].param[0]) = cTilePos;
        adf::location<adf::stack>(concat_graph.k1[i]) = cTilePos;
      }
    }

    QLinearConvChunkHPktStreamGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));
      static_assert(B*C*HCHUNK*PAD_W <= TILE_BYTES);

      adf::port<adf::input> pin0;
      adf::port<adf::input> pin1;
      pin.push_back(pin0);
      pin.push_back(pin1);

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, HCHUNK, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
          bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
        adf::connect<adf::stream>    (pin[1], k[i].in[1]);
      }
      init_helper(x_zero);
    }

    QLinearConvChunkHPktStreamGraph(
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));
      static_assert(B*C*HCHUNK*PAD_W <= TILE_BYTES);
      
      adf::port<adf::input> pin0;
      pin.push_back(pin0);

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<QLINEARCONV<TT, TTPARAM, HCHUNK, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
          weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      }
      init_helper(x_zero);
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by C dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_W*INP_W_PAD}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV0, 
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV1, 
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV2, 
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkCGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    typedef SplitFilterPktStreamGraph<SplitFilterInt8PktStream, TT, B, C*PAD_H*PAD_W, CCHUNK*PAD_H*PAD_W, 0> mSplitGraph;
    mSplitGraph split_graph;

    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    
  public:
    static constexpr int LCNT = C / CCHUNK;
    adf::kernel k[LCNT];

    adf::port<adf::input> pin[1];
    adf::port<adf::output> pout[1];

    QLinearConvChunkCGraph(
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) {
      static_assert(LCNT >= 3);
      static_assert(C % CCHUNK == 0);
      static_assert(B*CCHUNK*PAD_H*PAD_W <= TILE_BYTES);
      assert(weights.size() / LCNT <= TILE_BYTES); // weight size may vary based on padding done for given kernel

      for (int i = 0; i < LCNT; i++) {
        std::vector<TTPARAM> wChunk; // build wChunk
        wChunk.reserve(weights.size() / LCNT);
        for (int m = 0; m < M; m++) {
          wChunk.insert(wChunk.end(), 
            weights.begin() + m*weights.size()/M + i*weights.size()/M/LCNT, 
            weights.begin() + m*weights.size()/M + (i+1)*weights.size()/M/LCNT);
        }

        if (i == 0) {
          k[i] = adf::kernel::create_object<QLINEARCONV0<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, CCHUNK, M, KH, KW, GROUP>>(
            wChunk, bias, w_zero);
        } else if (i == LCNT-1) {
          k[i] = adf::kernel::create_object<QLINEARCONV2<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, CCHUNK, M, KH, KW, GROUP>>(
            wChunk, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
          adf::connect<adf::cascade> (k[i-1].out[0], k[i].in[1]);
          adf::connect<adf::stream>  (k[i].out[0], pout[0]);
        } else {
          k[i] = adf::kernel::create_object<QLINEARCONV1<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, CCHUNK, M, KH, KW, GROUP>>(
            wChunk, w_zero);
          adf::connect<adf::cascade> (k[i-1].out[0], k[i].in[1]);
        }
        
        adf::source(k[i]) = "qlinearconv.cc";
        adf::headers(k[i]) = {"qlinearconv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::single_buffer(k[i].in[0]);

        adf::connect<adf::window<B*CCHUNK*PAD_H*PAD_W>> (split_graph.pout[i], k[i].in[0]);

        if (i != 0) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=1});
        }
        adf::location<adf::buffer>(k[i].in[0]) = adf::location<adf::kernel>(k[i]);
        adf::location<adf::buffer>(k[i].in[0]) = {adf::offset(0)};
      }

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamInt8<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
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
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by C dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_W*INP_W_PAD}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV0, 
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV1, 
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV2, 
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkCStreamGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    typedef SplitFilterPktStreamGraph<SplitFilterInt8PktStream, TT, B, C*PAD_H*PAD_W, CCHUNK*PAD_H*PAD_W, 0> mSplitGraph;
    mSplitGraph split_graph;

    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    
  public:
    static constexpr int LCNT = C / CCHUNK;
    adf::kernel k[LCNT];

    adf::port<adf::input> pin[1+LCNT];
    adf::port<adf::output> pout[1];

    QLinearConvChunkCStreamGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) {
      static_assert(LCNT >= 3);
      static_assert(C % CCHUNK == 0);
      static_assert(B*CCHUNK*PAD_H*PAD_W <= TILE_BYTES);

      for (int i = 0; i < LCNT; i++) {

        if (i == 0) {
          k[i] = adf::kernel::create_object<QLINEARCONV0<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, CCHUNK, M, KH, KW, GROUP>>(
            bias, w_zero);
          set_heap_size<QLINEARCONV0,TT,TTPARAM,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,CCHUNK,M,KH,KW,GROUP>(k[0]);
        } else if (i == LCNT-1) {
          k[i] = adf::kernel::create_object<QLINEARCONV2<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, CCHUNK, M, KH, KW, GROUP>>(
            x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
          adf::connect<adf::cascade> (k[i-1].out[0], k[i].in[2]);
          adf::connect<adf::stream>  (k[i].out[0], pout[0]);
          set_heap_size<QLINEARCONV2,TT,TTPARAM,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,CCHUNK,M,KH,KW,GROUP>(k[0]);
        } else {
          k[i] = adf::kernel::create_object<QLINEARCONV1<TT, TTPARAM, PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, CCHUNK, M, KH, KW, GROUP>>(
            w_zero);
          adf::connect<adf::cascade> (k[i-1].out[0], k[i].in[2]);
          set_heap_size<QLINEARCONV1,TT,TTPARAM,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,CCHUNK,M,KH,KW,GROUP>(k[0]);
        }
        adf::connect<adf::stream>    (pin[1+i], k[i].in[1]);
        
        adf::source(k[i]) = "qlinearconv.cc";
        adf::headers(k[i]) = {"qlinearconv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;

        adf::connect<adf::pktstream> (split_graph.pout[i], k[i].in[0]);

        if (i != 0) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=1});
        }
        adf::location<adf::stack>(k[i]) = adf::location<adf::kernel>(k[i]);
      }

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamInt8<TT, B*C, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>>(x_zero));
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
    }

};
/** @} */


#endif // __QLINEARCONV_GRAPH_H__