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
 * @connect{pout[0], stream H*OUT_W*TTSIZE}
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

      for (int i = 0; i < LCNT; i++) {
        adf::connect<adf::window<H*INP_W*TTSIZE>> (pin[i], k[0].in[i]);
        adf::single_buffer(k[0].in[i]);
      }
      
      // OUT_W <= H*INP_W
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::samples_per_iteration(k[0].out[0]) = H*OUT_W;
    }

};


template <template<typename, int, int, int, int> class CONCAT_STREAM,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatStreamGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    static constexpr int L1_LCNT = LCNT / 2;
  
  public:
    adf::vector<adf::kernel> k1;
    adf::vector<adf::kernel> k2;
    adf::vector<adf::kernel> k;

    adf::port<input> pin[LCNT];
    adf::port<output> pout[1];

    template<int mINP_W1, int mINP_W2, int mOUT_W>
    adf::kernel create_concat_kernel() {
      adf::kernel new_k = adf::kernel::create_object<CONCAT_STREAM<TT, H, mINP_W1, mINP_W2, mOUT_W>>();
      adf::source(new_k) = "concat.cc";
      adf::headers(new_k) = {"concat.h"};
      adf::runtime<ratio>(new_k) = 0.1;
      return new_k;
    }

    // separate tiles since each tile only 2 input streams
    ConcatStreamGraph() {
      static_assert(LCNT <= 8 && LCNT > 1);

      adf::kernel _k;
      for (int i = 0; i < LCNT-1; i+=2) {
        _k = create_concat_kernel<INP_W, INP_W, 2*INP_W>();
        k1.push_back(_k);
        adf::connect<adf::stream> (pin[i], _k.in[0]);
        adf::connect<adf::stream> (pin[i+1], _k.in[1]);
      }
      for (int i = 0; i < LCNT-3; i+=4) {
        _k = create_concat_kernel<2*INP_W, 2*INP_W, 4*INP_W>();
        k2.push_back(_k);
        adf::connect<adf::stream> (k1[i/2].out[0], _k.in[0]);
        adf::connect<adf::stream> (k1[i/2+1].out[0], _k.in[1]);
      }


      if (LCNT == 2) {
        adf::connect<adf::stream> (k1[0].out[0], pout[0]);
        adf::samples_per_iteration(k1[0].out[0]) = H*OUT_W;
      }
      else if (LCNT == 3) {
        _k = create_concat_kernel<2*INP_W, INP_W, OUT_W>();
        k.push_back(_k);
        adf::connect<adf::stream> (k1[0].out[0], _k.in[0]);
        adf::connect<adf::stream> (pin[LCNT-1], _k.in[1]);

        adf::connect<adf::stream> (_k.out[0], pout[0]);
        adf::samples_per_iteration(_k.out[0]) = H*OUT_W;
      }
      else if (LCNT == 4) {
        adf::connect<adf::stream> (k2[0].out[0], pout[0]);
        adf::samples_per_iteration(k2[0].out[0]) = H*OUT_W;
      }
      else if (LCNT == 5) {
        _k = create_concat_kernel<4*INP_W, INP_W, OUT_W>();
        k.push_back(_k);
        adf::connect<adf::stream> (k2[0].out[0], _k.in[0]);
        adf::connect<adf::stream> (pin[LCNT-1], _k.in[1]);

        adf::connect<adf::stream> (_k.out[0], pout[0]);
        adf::samples_per_iteration(_k.out[0]) = H*OUT_W;
      }
      else if (LCNT == 6) {
        _k = create_concat_kernel<4*INP_W, 2*INP_W, OUT_W>();
        k.push_back(_k);
        adf::connect<adf::stream> (k2[0].out[0], _k.in[0]);
        adf::connect<adf::stream> (k1[2].out[0], _k.in[1]);

        adf::connect<adf::stream> (_k.out[0], pout[0]);
        adf::samples_per_iteration(_k.out[0]) = H*OUT_W;
      }
      else if (LCNT == 7) {
        adf::kernel _k1 = create_concat_kernel<2*INP_W, INP_W, 4*INP_W>();
        k.push_back(_k1);
        adf::connect<adf::stream> (k1[2].out[0], _k1.in[0]);
        adf::connect<adf::stream> (pin[LCNT-1], _k1.in[1]);

        adf::kernel _k2 = create_concat_kernel<4*INP_W, 4*INP_W, OUT_W>();
        k.push_back(_k2);
        adf::connect<adf::stream> (k2[0].out[0], _k2.in[0]);
        adf::connect<adf::stream> (_k1.out[0], _k2.in[1]);

        adf::connect<adf::stream> (_k2.out[0], pout[0]);
        adf::samples_per_iteration(_k2.out[0]) = H*OUT_W;
      }
      else if (LCNT == 8) {
        _k = create_concat_kernel<4*INP_W, 4*INP_W, OUT_W>();
        k.push_back(_k);
        adf::connect<adf::stream> (k2[0].out[0], _k.in[0]);
        adf::connect<adf::stream> (k2[1].out[0], _k.in[1]);

        adf::connect<adf::stream> (_k.out[0], pout[0]);
        adf::samples_per_iteration(_k.out[0]) = H*OUT_W;
      }
      
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
          adf::connect<adf::stream> (k[i].out[0], klast.in[i]);
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
      adf::connect<adf::stream> (k[i].out[0], klast.in[i]);
      
      adf::connect<adf::stream> (klast.out[0], pout[0]);
    }

};
/** @} */


#endif // __CONCAT_GRAPH_H__