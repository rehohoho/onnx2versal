#ifndef __POOL_GRAPH_H__
#define __POOL_GRAPH_H__

#include <adf.h>
#include "pool.h"


/**
 * @defgroup Pool2D
 * 
 * @brief Pool2D function on BCHW, yielding BCH'W', where H'=H/factor, W'=W/factor
 * 
 * @tparam POOL     Pool Kernel
 * @tparam INP_W    input width/height
 * @tparam OUT_W    output width/height, = INP_W - K/2
 * @tparam B        batch size
 * @tparam C        input channels
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*C*INP_W*INP_W*4}
 * @connect{pout[0], B*C*OUT_W*OUT_W*4}
 * @endconnections
 */
template <template<int, int, int, int> class POOL,
  int INP_W, int OUT_W, int B, int C>
class MaxpoolGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    MaxpoolGraph() { 
      k[0] = adf::kernel::create_object<POOL<INP_W, OUT_W, B, C>>();
      adf::source(k[0]) = "pool.cc";
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*C*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __POOL_GRAPH_H__