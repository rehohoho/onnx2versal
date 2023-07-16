#ifndef __QLINEARADD_GRAPH_H__
#define __QLINEARADD_GRAPH_H__

#include <adf.h>
#include "qlinearadd.h"


/**
 * @defgroup QLinearAdd
 * 
 * @brief QLinearAdd over W vector,
 * See https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add, 
 * See https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
 * 
 * @tparam QLINEARADD QLinearAdd Kernel
 * @tparam W		      width
 * @tparam IS_RELU    if RELU is done after qlinearadd
 * 
 * Computation
 * - qc = ((qa-qa_zero)*qa_scale + (qb-qb_zero)*qb_scale) / qc_scale + qc_zero
 *      = qa*qa_scale/qc_scale + qb*qb_scale/qc_scale + (-qa_zero*qa_scale - qb_zero*qb_scale)/qc_scale + qc_zero
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], stream W*TTSIZE}
 * @connect{pin[1], stream W*TTSIZE}
 * @connect{pout[0], stream W*TTSIZE}
 * @endconnections
 */
template <template<typename, int, int> class QLINEARADD, 
  typename TT, int W, int IS_RELU>
class QLinearAddGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[2];
    adf::port<output> pout[1];

    QLinearAddGraph(
      float a_scale,
			float b_scale,
			float c_scale,
			TT a_zero,
			TT b_zero,
			TT c_zero
    ) { 
      k[0] = adf::kernel::create_object<QLINEARADD<TT, W, IS_RELU>>(
        a_scale, b_scale, c_scale, a_zero, b_zero, c_zero);
      adf::source(k[0]) = "qlinearadd.cc";
      adf::headers(k[0]) = {"qlinearadd.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (pin[1], k[0].in[1]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);

      adf::samples_per_iteration(k[0].in[0]) = W;
      adf::samples_per_iteration(k[0].in[1]) = W;
      adf::samples_per_iteration(k[0].out[0]) = W;
    }

};
/** @} */


#endif // __QLINEARADD_GRAPH_H__