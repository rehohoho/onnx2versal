#ifndef __IDENTITY_GRAPH_H__
#define __IDENTITY_GRAPH_H__

#include <adf.h>
#include "identity.h"
#include "graph_utils.h"

/**
 * @defgroup Identity
 * 
 * @brief Identity function, output=input
 * 
 * @tparam IDENTITY Identity Kernel
 * @tparam N        size of input and output
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], stream N*sizeof(TT)}
 * @connect{pout[0], stream N*sizeof(TT)}
 * @endconnections
 */
template <typename TT, int N>
class IdentityGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    IdentityGraph() { 
      k[0] = adf::kernel::create_object<Identity<TT, N>>();
      adf::source(k[0]) = "identity.cc";
      adf::headers(k[0]) = {"identity.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::samples_per_iteration(k[0].out[0]) = N;
    }

};
/** @} */


#endif // __IDENTITY_GRAPH_H__