#ifndef __SOFTMAX_GRAPH_H__
#define __SOFTMAX_GRAPH_H__

#include <adf.h>
#include "softmax.h"


/**
 * @defgroup Softmax
 * 
 * @brief Softmax over CHUNK_CNT chunks of CHUNK_SIZE vector.
 * See https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
 * Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * 
 * @tparam SOFTMAX        Softmax Kernel
 * @tparam INP_H	        input height
 * @tparam INP_W	        input width
 * @tparam INP_W_PAD	    input width padded
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], INP_H*INP_W_PAD*4}
 * @connect{pout[0], INP_H*INP_W_PAD*4}
 * @endconnections
 */
template <template<int, int, int> class SOFTMAX,
  int INP_H, int INP_W, int INP_W_PAD>
class SoftmaxGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    SoftmaxGraph() { 
      k[0] = adf::kernel::create_object<SOFTMAX<INP_H, INP_W, INP_W_PAD>>();
      adf::source(k[0]) = "softmax.cc";
      adf::headers(k[0]) = {"softmax.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<INP_H*INP_W_PAD*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<INP_H*INP_W_PAD*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __SOFTMAX_GRAPH_H__