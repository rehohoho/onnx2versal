#ifndef __QLINEARSOFTMAX_GRAPH_H__
#define __QLINEARSOFTMAX_GRAPH_H__

#include <adf.h>
#include <pad.h>
#include "qlinearsoftmax.h"


/**
 * @defgroup QLinearSoftmax
 * 
 * @brief QLinearSoftmax over INP_H chunks of INP_W_PAD vector.
 * See https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
 * Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * 
 * @tparam QLINEARSOFTMAX QLinearSoftmax Kernel
 * @tparam TT	            int8_t or uint8_t
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
 * @connect{pin[0], INP_H*INP_W_PAD}
 * @connect{pout[0], INP_H*INP_W_PAD}
 * @endconnections
 */
template <
  template<typename, int, int, int, int, int, int, int, int> class PAD,
  template<typename, int, int, int> class QLINEARSOFTMAX, 
  typename TT, int INP_H, int INP_W, int INP_W_PAD>
class QLinearSoftmaxStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::vector<adf::kernel> pad;
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QLinearSoftmaxStreamGraph(
      float x_scale,
      float y_scale,
      TT x_zero,
      TT y_zero
    ) { 
      k[0] = adf::kernel::create_object<QLINEARSOFTMAX<TT, INP_H, INP_W, INP_W_PAD>>(
        x_scale, y_scale, x_zero, y_zero);
      adf::source(k[0]) = "qlinearsoftmax.cc";
      adf::headers(k[0]) = {"qlinearsoftmax.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      if (INP_W_PAD > INP_W) {
        pad.push_back(
          adf::kernel::create_object<PAD<TT, 1, INP_H, INP_W, INP_W_PAD, 0, 0, 0, INP_W_PAD-INP_W>>(x_zero));
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream, adf::window<INP_H*INP_W_PAD>> (pad[0].out[0], k[0].in[0]);
      } else {
        adf::connect<adf::window<INP_H*INP_W_PAD>> (pin[0], k[0].in[0]);
      }

      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::samples_per_iteration(k[0].out[0]) = INP_H*INP_W_PAD;
    }

};
/** @} */


#endif // __QLINEARSOFTMAX_GRAPH_H__