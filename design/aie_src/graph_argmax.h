#ifndef __ARGMAX_GRAPH_H__
#define __ARGMAX_GRAPH_H__

#include <adf.h>
#include "argmax.h"


/**
 * @defgroup Argmax
 * 
 * @brief Argmax over CHUNK_CNT chunks of CHUNK_SIZE_PAD vector
 * 
 * @tparam ARGMAX         Argmax Kernel
 * @tparam CHUNK_CNT	    number of chunks
 * @tparam CHUNK_SIZE		  size of chunk per iteration to calculate
 * @tparam CHUNK_SIZE_PAD size of chunk, padded to vector boundary
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], CHUNK_CNT*CHUNK_SIZE_PAD*4}
 * @connect{pout[0], CHUNK_CNT*4}
 * @endconnections
 */
template <template<int, int, int> class ARGMAX, int CHUNK_CNT, int CHUNK_SIZE, int CHUNK_SIZE_PAD>
class ArgmaxGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    ArgmaxGraph() { 
      k[0] = adf::kernel::create_object<ARGMAX<CHUNK_CNT, CHUNK_SIZE, CHUNK_SIZE_PAD>>();
      adf::source(k[0]) = "argmax.cc";
      adf::headers(k[0]) = {"argmax.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<CHUNK_CNT*CHUNK_SIZE_PAD*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<CHUNK_CNT*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __ARGMAX_GRAPH_H__