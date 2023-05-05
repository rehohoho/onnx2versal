#ifndef __POOL_GRAPH_H__
#define __POOL_GRAPH_H__

#include <adf.h>
#include "pool.h"


template <template<int, int, int, int> class POOL,
  int INP_W, int OUT_W, int B, int C>
class MaxpoolGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    void construct() { 
      k[0] = adf::kernel::create_object<POOL<INP_W, OUT_W, B, C>>();
      adf::source(k[0]) = "pool.cc";
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*C*4>> (k[0].out[0], pout[0]);
    }

};


#endif // __POOL_GRAPH_H__