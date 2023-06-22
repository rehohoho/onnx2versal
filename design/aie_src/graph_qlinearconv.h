#ifndef __QLINEARCONV_GRAPH_H__
#define __QLINEARCONV_GRAPH_H__

#include <adf.h>
#include "qlinearconv.h"
#include "pad.h"
#include "graph_utils.h"


/**
 * @defgroup QLinearConv
 * 
 * @brief The convolution operator consumes a quantized input tensor, its scale and zero point, 
 * a quantized filter, its scale and zero point, and output's scale and zero point, 
 * and computes the quantized output. 2D convolution on H, W dimensions of BCHW using kernels MCKK. 
 * Each c-th KxK kernel is applied on C dimension. This is done over M iterations to yield
 * MxHxW per instance. This is done over B iterations to yield B batches.
 * 
 * @tparam QLINEARCONV     Conv2D Kernel
 * @tparam INP_H    input height
 * @tparam INP_W    input width
 * @tparam OUT_W    output width, padded to vector boundary = pad(INP_W - K/2)
 * @tparam STEP_H   stride in height dimension
 * @tparam STEP_W   stride in width dimension
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
 * @connect{pin[0], B*C*INP_H*INP_W}
 * @connect{pout[0], B*M*OUT_H*OUT_W}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W,
  int B, int C, int M, int K,
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
    static constexpr int OUT_H = (PAD_H - K) / STEP_H + 1;

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
      static_assert(B*M*OUT_H*OUT_W <= MAX_PARAM_BYTES);
      
      k[0] = adf::kernel::create_object<QLINEARCONV<PAD_H, PAD_W, OUT_W, STEP_H, STEP_W, B, C, M, K>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DWindowScalar<int8_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::window<B*C*INP_H*INP_W>, adf::window<INP_H*INP_W>> (pin[0], pad[0].in[0]);
        adf::connect<adf::window<PAD_H*PAD_W>, adf::window<B*C*PAD_H*PAD_W>> (pad[0].out[0], k[0].in[0]);
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W>> (pin[0], k[0].in[0]);
      }

      adf::connect<adf::window<B*M*OUT_H*OUT_W>> (k[0].out[0], pout[0]);

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
 * @connect{pout[0], stream B*M*OUT_H*OUT_W}
 * @endconnections
 */
template <template<int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
  int B, int C, int M, int K, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

  public:
    static constexpr int OUT_H = (PAD_H - K) / STEP_H + 1;

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
      static_assert(B*M*OUT_H*OUT_W <= MAX_PARAM_BYTES);
      
      k[0] = adf::kernel::create_object<QLINEARCONV<PAD_H, PAD_W, OUT_W, STEP_H, STEP_W, B, C, M, K>>(
        bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<Pad2DWindowScalar<int8_t, B*C, INP_H, INP_W, H0, H1, W0, W1>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::window<B*C*INP_H*INP_W>, adf::window<INP_H*INP_W>> (pin[0], pad[0].in[0]);
        adf::connect<adf::window<PAD_H*PAD_W>, adf::window<B*C*PAD_H*PAD_W>> (pad[0].out[0], k[0].in[0]);

        adf::location<adf::buffer>(k[0].in[0]) = adf::location<adf::kernel>(k[0]);
      } else {
        adf::connect<adf::window<B*C*INP_H*INP_W>> (pin[0], k[0].in[0]);
      }
      
      adf::connect<adf::stream> (pin[1], k[0].in[1]); // variable samples per iteration based on kernel
      adf::connect<adf::window<B*M*OUT_H*OUT_W>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __QLINEARCONV_GRAPH_H__