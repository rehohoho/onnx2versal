#ifndef __QLINEARCONV_GRAPH_H__
#define __QLINEARCONV_GRAPH_H__

#include <adf.h>
#include "qlinearconv.h"


/**
 * @defgroup QLinearConv
 * 
 * @brief The convolution operator consumes a quantized input tensor, its scale and zero point, 
 * a quantized filter, its scale and zero point, and output's scale and zero point, 
 * and computes the quantized output. 2D convolution on H, W dimensions of BCHW using kernels MCKK. 
 * Each c-th KxK kernel is applied on C dimension. This is done over M iterations to yield
 * MxHxW per instance. This is done over B iterations to yield B batches.
 * 
 * @tparam CONV     Conv2D Kernel
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
 * @connect{pin[0], B*C*INP_W*INP_W}
 * @connect{pout[0], B*M*OUT_W*OUT_W}
 * @endconnections
 */
template <template<int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K>
class QLinearConvGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QLinearConvGraph(
      std::vector<float> weights,
      std::vector<float> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero_point,
      int8_t w_zero_point,
      int8_t y_zero_point
    ) { 
      k[0] = adf::kernel::create_object<CONV<INP_W, OUT_W, B, C, M, K>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero_point, w_zero_point, y_zero_point);
      adf::source(k[0]) = "qlinearconv.cc";
      adf::headers(k[0]) = {"qlinearconv.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::window<B*INP_W*INP_W*C>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M>> (k[0].out[0], pout[0]);

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos; // weight (<= 16384B)
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0x0000);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos; // bias   (<= 4096B)
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset(0x4000); 
    }

};
/** @} */


#endif // __QLINEARCONV_GRAPH_H__