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
 * @tparam TT          input, output data type, int8_t or uint8_t only
 * @tparam TTPARAM     weight, bias data type, int8_t or uint8_t only
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
template <
  template<typename, typename, int, int, int> class QLINEARMAC, 
  typename TT, typename TTPARAM, int B, int W, int IS_RELU>
class QlinearMacGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QlinearMacGraph(
      std::vector<TTPARAM> weights,
      std::vector<TTPARAM> bias,
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TTPARAM b_zero,
      TT z_zero,
      TT y_zero,
      int repeat_cnt = 1
    ) { 
      static_assert(2*W < MAX_PARAM_BYTES);
      k[0] = adf::kernel::create_object<QLINEARMAC<TT, TTPARAM, B, W, IS_RELU>>(
        weights, bias, x_scale, w_scale, b_scale, z_scale, y_scale, x_zero, w_zero, b_zero, z_zero, y_zero);
      adf::source(k[0]) = "qlinearmac.cc";
      adf::headers(k[0]) = {"qlinearmac.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      adf::heap_size(k[0]) = 3*W*4 + 1024;
      
      adf::connect<adf::window<B*W>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*W>> (k[0].out[0], pout[0]);

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset((W+31)/32*32); 
    }

};


/**
 * @brief Single instance stream graph
 * 
 * @connections
 * @connect{pin[0], B*W}
 * @connect{pout[0], B*W}
 * @endconnections
 */
template <
  template<typename, typename, int, int, int> class QLINEARMAC, 
  typename TT, typename TTPARAM, int B, int W, int IS_RELU>
class QlinearMacStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QlinearMacStreamGraph(
      std::vector<TTPARAM> weights,
      std::vector<TTPARAM> bias,
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TTPARAM b_zero,
      TT z_zero,
      TT y_zero
    ) { 
      static_assert(2*W < MAX_PARAM_BYTES);
      k[0] = adf::kernel::create_object<QLINEARMAC<TT, TTPARAM, B, W, IS_RELU>>(
        weights, bias, x_scale, w_scale, b_scale, z_scale, y_scale, x_zero, w_zero, b_zero, z_zero, y_zero);
      adf::source(k[0]) = "qlinearmac.cc";
      adf::headers(k[0]) = {"qlinearmac.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::heap_size(k[0]) = 3*W*4 + 1024;
      
      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);

      adf::samples_per_iteration(k[0].in[0]) = B*W;
      adf::samples_per_iteration(k[0].out[0]) = B*W;

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
      adf::location<adf::parameter>(k[0].param[1]) = tilePos;
      adf::location<adf::parameter>(k[0].param[1]) = adf::offset((W+31)/32*32); 
    }

};
/** @} */


#endif // __QLINEARMAC_GRAPH_H__