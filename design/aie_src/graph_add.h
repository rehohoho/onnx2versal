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
 * @connect{pin[0], W*TTSIZE}
 * @connect{pout[0], W*TTSIZE}
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

    AddGraph(
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<ADD<TT, W, IS_RELU>>();
      adf::source(k[0]) = "add.cc";
      adf::headers(k[0]) = {"add.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      
      adf::connect<adf::window<W*TTSIZE>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<W*TTSIZE>> (pin[1], k[0].in[1]);
      adf::connect<adf::window<W*TTSIZE>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __ADD_GRAPH_H__