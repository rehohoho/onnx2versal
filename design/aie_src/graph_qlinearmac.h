#ifndef __QLINEARMAC_GRAPH_H__
#define __QLINEARMAC_GRAPH_H__

#include <assert.h>
#include <adf.h>
#include "qlinearmac.h"
#include "graph_utils.h"


/**
 * @defgroup QlinearMac
 * 
 * @brief 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
 * - y = saturate ((x / y_scale) + y_zero)
 * - x = (qx - qx_zero) * qx_scale
 * 
 * @tparam QLINEARMAC  QlinearMac Kernel
 * @tparam B           batch
 * @tparam W           width
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*W}
 * @connect{pout[0], B*W}
 * @endconnections
 */
template <template<int, int, int> class QLINEARMAC, 
  int B, int W, int IS_RELU>
class QlinearMacGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QlinearMacGraph(
      std::vector<int8_t> weights,
      std::vector<int8_t> bias,
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t b_zero,
      int8_t z_zero,
      int8_t y_zero,
      int repeat_cnt = 1
    ) { 
      static_assert(2*W < MAX_PARAM_BYTES);
      k[0] = adf::kernel::create_object<QLINEARMAC<B, W, IS_RELU>>(
        weights, bias, x_scale, w_scale, b_scale, z_scale, y_scale, x_zero, w_zero, b_zero, z_zero, y_zero);
      adf::source(k[0]) = "qlinearmac.cc";
      adf::headers(k[0]) = {"qlinearmac.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      adf::heap_size(k[0]) = 2*W*4 + 1024;
      
      adf::connect<adf::window<B*W>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*W>> (k[0].out[0], pout[0]);

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset((W+31)/32*32); 
    }

};
/** @} */


#endif // __QLINEARMAC_GRAPH_H__