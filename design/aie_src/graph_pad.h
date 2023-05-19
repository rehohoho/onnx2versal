#ifndef __PAD_GRAPH_H__
#define __PAD_GRAPH_H__

#include <adf.h>
#include "pad.h"


/**
 * @defgroup Pad
 * 
 * @brief Pad graphs. Pads NxINP_W to NxOUT_W shape
 * 
 * @tparam PAD    Pad Kernel
 * @tparam TT     Data type
 * @tparam N      Number of rows
 * @tparam INP_W  Length of input row
 * @tparam OUT_W  Length of output row
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], N*INP_W*sizeof(TT)}
 * @connect{pout[0], N*OUT_W*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int> class PAD, 
  typename TT, int N, int INP_W, int OUT_W>
class PadGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    PadGraph() { 
      k[0] = adf::kernel::create_object<PAD<TT, N, INP_W, OUT_W>>();
      adf::source(k[0]) = "pad.cc";
      adf::headers(k[0]) = {"pad.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<N*INP_W*sizeof(TT)>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<N*OUT_W*sizeof(TT)>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __PAD_GRAPH_H__