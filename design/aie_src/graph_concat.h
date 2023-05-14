#ifndef __CONCAT_GRAPH_H__
#define __CONCAT_GRAPH_H__

#include <adf.h>
#include "concat.h"

/**
 * @defgroup Concat
 * 
 * @brief Concatenates chunks of CHUNK_SIZE from LCNT lanes then truncate to BLOCK_SIZE
 * 
 * @details
 * - Contains functions filter[k], k=1,2,...,8, registers filter based on LCNT
 * - Maximum of 8 lanes since max incoming DMA to a tile is 8
 * - Using virtual function instead of macro has big overhead: 163 -> 1047
 * 
 * @tparam CONCAT       Concat Kernel
 * @tparam LCNT 				number of lanes to concat
 * @tparam WINDOW_SIZE	size of window for each lane
 * @tparam CHUNK_SIZE		size of chunk from each lanes per iteration
 * @tparam BLOCK_SIZE		size of concatenated chunks per iteration
 * 
 * @connections
 * @connect{pin[LCNT], WINDOW_SIZE*4}
 * @connect{pout[1], WINDOW_SIZE/CHUNK_SIZE*BLOCK_SIZE*4}
 * @endconnections
 * 
 * @attention ConcatVector breaks if CONCAT_CHUNK%8!=0 CONCAT_BLOCK%4!=0
 * 
 * @{
 */

/**
 * @brief Graph wrapper for arbitrary concat kernel implementation and lanes
 */
template <template<int, int, int, int> class CONCAT,
  int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class ConcatGraph : public adf::graph {

  public:
    adf::kernel k[1];
    adf::port<input> pin[LCNT];
    adf::port<output> pout[1];

    ConcatGraph() { 
      k[0] = adf::kernel::create_object<CONCAT<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      adf::source(k[0]) = "concat.cc";
      adf::runtime<ratio>(k[0]) = 0.6;

      for (int i = 0; i < LCNT; i++)
        adf::connect<adf::window<WINDOW_SIZE*4>> (pin[i], k[0].in[i]);
      
      // BLOCK_SIZE <= WINDOW_SIZE
      adf::connect<adf::window<WINDOW_SIZE/CHUNK_SIZE*BLOCK_SIZE*4>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __CONCAT_GRAPH_H__