#ifndef __QLINEARCONV_GRAPH_H__
#define __QLINEARCONV_GRAPH_H__

#include <adf.h>
#include "qlinearconv.h"
#include "pad.h"
#include "graph_concat.h"
#include "graph_split.h"
#include "graph_utils.h"


template <template<int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP>
void set_heap_size(adf::kernel k) {
  if (
    (std::is_same<
    QLINEARCONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConvScalarStream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConv3x3Stream<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConv3x3StreamPad<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value) ||
    (std::is_same<
    QLINEARCONV<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>, 
    QLinearConv3x3StreamScale32bit<INP_H,INP_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>>::value)
  ) {
    adf::heap_size(k) = C*16 + 1024; // caches CKK weights
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
 * @tparam INP_H        input height
 * @tparam INP_W        input width
 * @tparam OUT_W        output width, unable to infer due to width padding
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
 * @connect{pin[0], B*C*INP_H*INP_W}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, int, int, int, int, int, int, int> class PAD,
  template<int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
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
      std::vector<int8_t> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    ) { 
      static_assert(B*C*PAD_H*PAD_W <= MAX_PARAM_BYTES);
      assert(weights.size() <= MAX_PARAM_BYTES);
      static_assert(B*M*OUT_H*OUT_W_PAD <= MAX_PARAM_BYTES);
      
      k[0] = adf::kernel::create_object<QLINEARCONV<PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<PAD<int8_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W>> (pad[0].out[0], k[0].in[0]);
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W>> (pin[0], k[0].in[0]);
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
 * @connect{pin[0], B*C*INP_H*INP_W}
 * @connect{pin[1], stream}
 * @connect{pout[0], stream B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, int, int, int, int, int, int, int> class PAD,
  template<int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
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

    adf::port<input> pin[2];
    adf::port<output> pout[1];

    QLinearConvStreamGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    ) { 
      static_assert(B*C*PAD_H*PAD_W <= MAX_PARAM_BYTES);
      
      k[0] = adf::kernel::create_object<QLINEARCONV<PAD_H, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
        bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      set_heap_size<QLINEARCONV,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>(k[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<PAD<int8_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<B*C*PAD_H*PAD_W>> (pad[0].out[0], k[0].in[0]);

        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;

        adf::location<adf::buffer>(k[0].in[0]) = adf::location<adf::kernel>(k[0]);
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::stream> (pin[1], k[0].in[1]); // variable samples per iteration based on kernel
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::samples_per_iteration(k[0].out[0]) = B*M*OUT_H*OUT_W_PAD;
      
      adf::location<adf::buffer>(k[0].in[0]) = adf::location<adf::kernel>(k[0]);
    }

};


/**
 * @brief Multiinstance graph that stores weights and biases, 
 * chunks BCHW by H dimension, maximum 8 chunks
 * 
 * @connections
 * @connect{pin[0], stream B*C*INP_W*INP_W}
 * @connect{pout[0], B*M*OUT_H*OUT_W_PAD}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    std::vector<adf::kernel> pad;

    static constexpr int OVERLAP = KH-STEP_H;
    typedef SplitGraph<SPLIT, int8_t, B*C, PAD_H*PAD_W, HCHUNK*PAD_W, OVERLAP*PAD_W> mSplitGraph;
    mSplitGraph split_graph;
    static constexpr int LCNT = mSplitGraph::LCNT;

    adf::kernel k[LCNT];

    static constexpr int HCHUNK_OUT = (HCHUNK - KH) / STEP_H + 1;
    static constexpr int OUT_H = (PAD_H - KH) / STEP_H + 1;
    ConcatStreamGraph<CONCAT, int8_t, LCNT, B*M, HCHUNK_OUT*OUT_W_PAD, OUT_H*OUT_W_PAD> concat_graph;

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
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    ) {
      static_assert((HCHUNK % STEP_H) == (KH % STEP_H));
      static_assert(LCNT <= 8);
      static_assert(B*C*PAD_H*PAD_W + B*C*(KH-1)*7*PAD_W <= MAX_PARAM_BYTES*8);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DStreamInt8<int8_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream> (pad[0].out[0], split_graph.pin[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
        // split and pad can't be placed on same tile due to stream co-placement constraints
      } else {
        adf::connect<adf::stream> (pin[0], split_graph.pin[0]);
      }

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<QLINEARCONV<HCHUNK, PAD_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>>(
          bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
        adf::source(k[i]) = "qlinearconv.cc";
        adf::headers(k[i]) = {"qlinearconv.h"};
        adf::runtime<ratio>(k[i]) = 0.6;

        set_heap_size<QLINEARCONV,PAD_H,PAD_W,OUT_W,OUT_W_PAD,STEP_H,STEP_W,B,C,M,KH,KW,GROUP>(k[i]);

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
/** @} */


#endif // __QLINEARCONV_GRAPH_H__