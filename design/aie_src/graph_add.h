#ifndef __ADD_GRAPH_H__
#define __ADD_GRAPH_H__

#include <adf.h>
#include "add.h"


/**
 * @defgroup Add
 * 
 * @brief Add over W vector,
 * see https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * 
 * @tparam ADD      Add Kernel
 * @tparam W		    width
 * @tparam IS_RELU  if RELU is done after add
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
template <template<typename, int, int> class ADD, 
  typename TT, int W, int IS_RELU>
class AddGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[2];
    adf::port<output> pout[1];

    AddGraph() { 
      k[0] = adf::kernel::create_object<ADD<TT, W, IS_RELU>>();
      adf::source(k[0]) = "add.cc";
      adf::headers(k[0]) = {"add.h"};
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


#endif // __ADD_GRAPH_H__