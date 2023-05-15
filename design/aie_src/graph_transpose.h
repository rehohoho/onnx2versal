#ifndef __TRANSPOSE_GRAPH_H__
#define __TRANSPOSE_GRAPH_H__

#include <adf.h>
#include "transpose.h"


/**
 * @defgroup Transpose
 * 
 * @brief Transpose function, BHWC->BCHW tentatively
 * 
 * @tparam TRANSPOSE  Transpose Kernel
 * @tparam B          batch size
 * @tparam H          height
 * @tparam W          width
 * @tparam C          input channels
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*H*W*C*4}
 * @connect{pout[0], B*H*W*C*4}
 * @endconnections
 */
template <template<int, int, int, int> class TRANSPOSE, 
  int B, int H, int W, int C>
class TransposeGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    TransposeGraph() { 
      k[0] = adf::kernel::create_object<TRANSPOSE<B, H, W, C>>();
      adf::source(k[0]) = "transpose.cc";
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*H*W*C*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*C*H*W*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __TRANSPOSE_GRAPH_H__