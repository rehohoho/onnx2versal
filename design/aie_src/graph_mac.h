#ifndef __MAC_GRAPH_H__
#define __MAC_GRAPH_H__

#include <assert.h>
#include <adf.h>
#include "mac.h"
#include "graph_utils.h"


/**
 * @defgroup Mac
 * 
 * @brief 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
 * 
 * @tparam MAC  Mac Kernel
 * @tparam B    batch
 * @tparam W    width
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*W*TTSIZE}
 * @connect{pout[0], B*W*TTSIZE}
 * @endconnections
 */
template <template<typename, int, int, int> class MAC, 
  typename TT, int B, int W, int IS_RELU>
class MacGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    MacGraph(
      std::vector<TT> weights,
      std::vector<TT> bias,
      int repeat_cnt = 1
    ) { 
      static_assert(W*TTSIZE < MAX_PARAM_BYTES);
      k[0] = adf::kernel::create_object<MAC<TT, B, W, IS_RELU>>(weights, bias);
      adf::source(k[0]) = "mac.cc";
      adf::headers(k[0]) = {"mac.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      
      adf::connect<adf::window<B*W*TTSIZE>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*W*TTSIZE>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __MAC_GRAPH_H__