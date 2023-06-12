#ifndef __POOL_GRAPH_H__
#define __POOL_GRAPH_H__

#include <adf.h>
#include "pool.h"


/**
 * @defgroup Pool2D
 * 
 * @brief Pool2D function on BCHW, yielding BCH'W', where H'=H/factor, W'=W/factor
 * Scalar kernels allow W'<W/factor.
 * 
 * @tparam TT         input and output type
 * @tparam POOL       Pool Kernel
 * @tparam INP_H      input height, used to calculate pool factor
 * @tparam INP_W      input width
 * @tparam OUT_W      output height, used to calculate pool factor
 * @tparam OUT_W      output width
 * @tparam B          batch size
 * @tparam C          input channels
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W*C*TTSIZE}
 * @connect{pout[0], B*OUT_H*OUT_W*C*TTSIZE}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int> class POOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
class PoolGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    static constexpr int K = INP_H / OUT_H;
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    PoolGraph() { 
      k[0] = adf::kernel::create_object<POOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>>();
      adf::source(k[0]) = "pool.cc";
      adf::headers(k[0]) = {"pool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_H*INP_W*C*TTSIZE>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_H*OUT_W*C*TTSIZE>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __POOL_GRAPH_H__