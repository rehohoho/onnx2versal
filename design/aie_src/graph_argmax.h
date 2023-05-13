#ifndef __ARGMAX_GRAPH_H__
#define __ARGMAX_GRAPH_H__

#include <adf.h>
#include "argmax.h"


/**
 * @defgroup Argmax
 * 
 * @brief Argmax over chunks of CHUNK_SIZE in WINDOW_SIZE vector
 * 
 * @tparam ARGMAX       Argmax Kernel
 * @tparam WINDOW_SIZE	size of window
 * @tparam CHUNK_SIZE		size of chunk per iteration
 * 
 * @connections
 * @connect{pin[1], WINDOW_SIZE*4}
 * @connect{pout[1], WINDOW_SIZE/CHUNK_SIZE*4}
 * @endconnections
 * 
 * @{
 */
template <template<int, int> class ARGMAX, int WINDOW_SIZE, int CHUNK_SIZE>
class ArgmaxGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ArgmaxGraph() { 
      k[0] = adf::kernel::create_object<ARGMAX<WINDOW_SIZE, CHUNK_SIZE>>();
      adf::source(k[0]) = "argmax.cc";
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<WINDOW_SIZE*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<WINDOW_SIZE/CHUNK_SIZE*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __ARGMAX_GRAPH_H__