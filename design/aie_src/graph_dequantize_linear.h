#ifndef __DEQUANTIZE_LINEAR_GRAPH_H__
#define __DEQUANTIZE_LINEAR_GRAPH_H__

#include <assert.h>
#include <adf.h>
#include "dequantize_linear.h"


/**
 * @defgroup DequantizeLinear
 * 
 * @brief Linear dequantization operator. It consumes a quantized tensor, a scale, and 
 * a zero point to compute the full precision tensor. The dequantization formula is 
 * y = (x - x_zero_point) * x_scale. x_scale and x_zero_point must have same shape, 
 * and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor 
 * for per-axis quantization. x_zero_point and x must have same type. x and y must 
 * have same shape.
 * 
 * @tparam DEQUANTIZE_LINEAR  DequantizeLinear Kernel
 * @tparam TT                 int8_t or uint8_t
 * @tparam B                  batch size
 * @tparam INP_W              input width
 * @tparam OUT_W              output width, expect OUT_W > INP_W
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_W}
 * @connect{pout[0], B*OUT_W*4}
 * @endconnections
 */
template <template<typename, int, int, int> class DEQUANTIZE_LINEAR, 
  typename TT, int B, int INP_W, int OUT_W>
class DequantizeLinearGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    DequantizeLinearGraph(
      float scale,
      TT zero,
      int repeat_cnt = 1
    ) { 
      static_assert(INP_W >= OUT_W);
      k[0] = adf::kernel::create_object<DEQUANTIZE_LINEAR<TT, B, INP_W, OUT_W>>(scale, zero);
      adf::source(k[0]) = "dequantize_linear.cc";
      adf::headers(k[0]) = {"dequantize_linear.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      
      adf::connect<adf::window<B*INP_W>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_W*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __DEQUANTIZE_LINEAR_GRAPH_H__