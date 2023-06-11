#ifndef __CONCAT_GRAPH_H__
#define __CONCAT_GRAPH_H__

#include <assert.h>
#include <adf.h>
#include "concat.h"

/**
 * @defgroup Concat
 * 
 * @brief Concatenates chunks of H*INP_W from LCNT lanes then truncate to H*OUT_W.
 * 
 * @details
 * - Contains functions filter[k], k=1,2,...,8, registers filter based on LCNT
 * - Maximum of 8 lanes since max incoming DMA to a tile is 8
 * - Using virtual function instead of macro has big overhead: 163 -> 1047
 * 
 * @tparam CONCAT   Concat Kernel
 * @tparam LCNT 		number of lanes to concat
 * @tparam H	      number of chunks
 * @tparam INP_W		size of chunk from each lanes per iteration
 * @tparam OUT_W		size of concatenated chunks per iteration
 * 
 * @{
 */

/**
 * @brief Graph wrapper for arbitrary concat kernel implementation and lanes
 * 
 * @connections
 * @connect{pin[0:LCNT], H*INP_W*TTSIZE}
 * @connect{pout[0], H*OUT_W*TTSIZE}
 * @endconnections
 */
template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatGraph : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
  
  public:
    adf::kernel k[1];
    adf::port<input> pin[LCNT];
    adf::port<output> pout[1];

    ConcatGraph() { 
      static_assert(LCNT <= 8);
      k[0] = adf::kernel::create_object<CONCAT<TT, LCNT, H, INP_W, OUT_W>>();
      adf::source(k[0]) = "concat.cc";
      adf::headers(k[0]) = {"concat.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      for (int i = 0; i < LCNT; i++)
        adf::connect<adf::window<H*INP_W*TTSIZE>> (pin[i], k[0].in[i]);
      
      // OUT_W <= H*INP_W
      adf::connect<adf::window<H*OUT_W*TTSIZE>> (k[0].out[0], pout[0]);
    }

};


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatTwiceGraph : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
  
  public:
    static constexpr int CONCAT_CNT = (LCNT+7)/8;
    static constexpr int LCNT_REM = (LCNT % 8 == 0) ? 8 : LCNT % 8;
    static constexpr int INNER_OUT_W = INP_W * 8;
    adf::kernel k[CONCAT_CNT];
    adf::kernel klast;
    adf::port<input> pin[LCNT];
    adf::port<output> pout[1];

    ConcatTwiceGraph() { 
      static_assert(LCNT <= 64);
      klast = adf::kernel::create_object<CONCAT<TT, CONCAT_CNT, H, INNER_OUT_W, OUT_W>>();
      adf::source(klast) = "concat.cc";
      adf::headers(klast) = {"concat.h"};
      adf::runtime<ratio>(klast) = 0.6;

      // intermediate concats
      if (CONCAT_CNT > 1) { // register kernel error if CONCAT_CNT=1
        for (int i = 0; i < CONCAT_CNT - 1; i++) {
          k[i] = adf::kernel::create_object<CONCAT<TT, 8, H, INP_W, INNER_OUT_W>>();
          adf::source(k[i]) = "concat.cc";
          adf::headers(k[i]) = {"concat.h"};
          adf::runtime<ratio>(k[i]) = 0.6;

          for (int j = 0; j < 8; j++)
            adf::connect<adf::window<H*INP_W*TTSIZE>> (pin[i*8+j], k[i].in[j]);
          adf::connect<adf::window<H*INNER_OUT_W*TTSIZE>> (k[i].out[0], klast.in[i]);
        }
      }
      
      int i = CONCAT_CNT - 1;
      // remainder intermediate concat
      k[i] = adf::kernel::create_object<CONCAT<TT, LCNT_REM, H, INP_W, INNER_OUT_W>>();
      adf::source(k[i]) = "concat.cc";
      adf::headers(k[i]) = {"concat.h"};
      adf::runtime<ratio>(k[i]) = 0.6;
      
      for (int j = 0; j < LCNT_REM; j++)
        adf::connect<adf::window<H*INP_W*TTSIZE>> (pin[i*8+j], k[i].in[j]);
      adf::connect<adf::window<H*INNER_OUT_W*TTSIZE>> (k[i].out[0], klast.in[i]);
      
      adf::connect<adf::window<H*OUT_W*TTSIZE>> (klast.out[0], pout[0]);
    }

};
/** @} */


#endif // __CONCAT_GRAPH_H__