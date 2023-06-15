#ifndef __PAD_GRAPH_H__
#define __PAD_GRAPH_H__

#include <assert.h>
#include <adf.h>
#include "pad.h"


/**
 * @defgroup Pad
 * 
 * @brief See padding at https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv for Pad2D
 * 
 * @tparam PAD    Pad Kernel
 * @tparam TT     Data type
 * @tparam INP_H  Input height
 * @tparam INP_W  Input width
 * @tparam H0     Pixels added before height
 * @tparam H1     Pixels added after height
 * @tparam W0     Pixels added before width
 * @tparam W1     Pixels added after width
 * 
 * @{
 */

/**
 * @brief Single instance graph for Pad2D
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W*sizeof(TT)}
 * @connect{pout[0], B*OUT_H*OUT_W*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int, int> class PAD, 
  typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
class Pad2DGraph : public adf::graph {

  private:
    static constexpr int OUT_H = INP_H + H0 + H1;
    static constexpr int OUT_W = INP_W + W0 + W1;
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    Pad2DGraph() { 
      static_assert(H0 >= 0 && H1 >= 0 && W0 >= 0 && W1 >= 0);
      k[0] = adf::kernel::create_object<PAD<TT, B, INP_H, INP_W, H0, H1, W0, W1>>();
      adf::source(k[0]) = "pad.cc";
      adf::headers(k[0]) = {"pad.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __PAD_GRAPH_H__