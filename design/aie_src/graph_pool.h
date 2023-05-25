#ifndef __POOL_GRAPH_H__
#define __POOL_GRAPH_H__

#include <adf.h>
#include "pool.h"


/**
 * @defgroup Pool2D
 * 
 * @brief Pool2D function on BCHW, yielding BCH'W', where H'=H/factor, W'=W/factor
 * 
 * @tparam TT         input and output type
 * @tparam POOL       Pool Kernel
 * @tparam INP_H      input height
 * @tparam INP_W      input width for computation
 * @tparam INP_W_PAD  input width of tensor
 * @tparam OUT_W      output width for computation = INP_W - K/2
 * @tparam OUT_W_PAD  output width of tensor
 * @tparam B          batch size
 * @tparam C          input channels
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W_PAD*C*TTSIZE}
 * @connect{pout[0], B*OUT_W*OUT_W_PAD*C*TTSIZE}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int, int> class POOL,
  typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int B, int C>
class MaxpoolGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    static constexpr int K = INP_H/OUT_W;
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    MaxpoolGraph() { 
      k[0] = adf::kernel::create_object<POOL<TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C>>();
      adf::source(k[0]) = "pool.cc";
      adf::headers(k[0]) = {"pool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_H*INP_W_PAD*C*TTSIZE>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*INP_H/K*OUT_W_PAD*C*TTSIZE>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __POOL_GRAPH_H__