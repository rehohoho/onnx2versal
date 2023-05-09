#ifndef __IDENTITY_GRAPH_H__
#define __IDENTITY_GRAPH_H__

#include <adf.h>
#include "identity.h"


template <template<int> class IDENTITY, int N>
class IdentityGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    IdentityGraph() { 
      k[0] = adf::kernel::create_object<IDENTITY<N>>();
      adf::source(k[0]) = "identity.cc";

      adf::connect<adf::window<N*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<N*4>> (k[0].out[0], pout[0]);
      adf::runtime<ratio>(k[0]) = 0.6;
    }

};


#endif // __IDENTITY_GRAPH_H__