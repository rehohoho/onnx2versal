#ifndef __QUANTIZE_LINEAR_GRAPH_H__
#define __QUANTIZE_LINEAR_GRAPH_H__

#include <adf.h>
#include "quantize_linear.h"


/**
 * @defgroup QuantizeLinear
 * 
 * @brief Linear quantization operator. It consumes a high precision tensor, a scale, and 
 * a zero point to compute the low precision / quantized tensor. The quantization formula 
 * is y = saturate ((x / y_scale) + y_zero). For saturation, it saturates to [0, 255] 
 * if it's uint8, or [-128, 127] if it's int8. For (x / y_scale), it's rounding to the 
 * nearest even. 
 * 
 * @tparam QUANTIZE_LINAER  QuantizeLinaer Kernel
 * @tparam INP_H	          input height
 * @tparam INP_W	          input width
 * @tparam OUT_W	          output width, allows padding, expects OUT_W >= INP_W
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], INP_H*INP_W*4}
 * @connect{pout[0], INP_H*OUT_W}
 * @endconnections
 */
template <template<int, int, int> class QUANTIZE_LINEAR, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QuantizeLinearGraph(
      float y_scale,
      int8_t y_zero,
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<QUANTIZE_LINEAR<INP_H, INP_W, OUT_W>>(y_scale, y_zero);
      adf::source(k[0]) = "quantize_linear.cc";
      adf::headers(k[0]) = {"quantize_linear.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      
      adf::connect<adf::window<INP_H*INP_W*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<INP_H*OUT_W>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __QUANTIZE_LINEAR_GRAPH_H__