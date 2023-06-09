#ifndef __SPLIT_GRAPH_H__
#define __SPLIT_GRAPH_H__

#include <assert.h>
#include <adf.h>
#include "split.h"

/**
 * @defgroup Split
 * 
 * @brief Splits width dimension of HxINP_W into split times HxOUT_W with optional overlap
 * 
 * @details
 * - Contains functions filter[k], k=1,2,...,8, registers filter based on LCNT
 * - Maximum of 8 lanes since max incoming DMA to a tile is 8
 * 
 * @tparam SPLIT    Split Kernel
 * @tparam TT       input/output type
 * @tparam H        input/output height
 * @tparam INP_W    input width
 * @tparam OUT_W    output width of each split
 * @tparam OVERLAP  number of overlapping elements between adjacent OUT_W
 * 
 * @{
 */

/**
 * @brief Graph wrapper for arbitrary split kernel implementation and lanes
 * 
 * @connections
 * @connect{pin[0], stream H*INP_W*sizeof(TT)}
 * @connect{pout[0:LCNT], H*OUT_W*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitGraph : public adf::graph {

  public:
		static constexpr int LCNT = (INP_W - OUT_W) / (OUT_W - OVERLAP) + 1;

    adf::kernel k[1];
    adf::port<input> pin[1];
    adf::port<output> pout[LCNT];

    SplitGraph() { 
      static_assert(LCNT <= 8);
      k[0] = adf::kernel::create_object<SPLIT<TT, H, INP_W, OUT_W, OVERLAP>>();
      adf::source(k[0]) = "split.cc";
      adf::headers(k[0]) = {"split.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::samples_per_iteration(k[0].in[0]) = H*INP_W;
      
      for (int i = 0; i < LCNT; i++) {
        adf::connect<adf::window<H*OUT_W*sizeof(TT)>> (k[0].out[i], pout[i]);
        adf::single_buffer(k[0].out[i]);
      }
    }

};

/**
 * @brief Graph wrapper for two stream split
 * 
 * @connections
 * @connect{pin[0], stream H*INP_W*sizeof(TT)}
 * @connect{pout[0], stream H*INP_W*sizeof(TT)* ((LCNT+1)/2)}
 * @connect{pout[1], stream H*INP_W*sizeof(TT)* (LCNT/2)}
 * @endconnections
 */
template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitTwoStreamGraph : public adf::graph {

  public:
		static constexpr int LCNT = (INP_W - OUT_W) / (OUT_W - OVERLAP) + 1;
    adf::kernel k[1];
    adf::port<input> pin[1];
    adf::port<output> pout[2];

    SplitTwoStreamGraph() { 
      k[0] = adf::kernel::create_object<SPLIT<TT, H, INP_W, OUT_W, OVERLAP>>();
      adf::source(k[0]) = "split.cc";
      adf::headers(k[0]) = {"split.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::connect<adf::stream> (k[0].out[1], pout[1]);
      
      adf::samples_per_iteration(k[0].in[0]) = H*INP_W;
      adf::samples_per_iteration(k[0].out[0]) = H*OUT_W* ((LCNT+1)/2);
      adf::samples_per_iteration(k[0].out[1]) = H*OUT_W* (LCNT/2);
    }

};


/**
 * @brief Graph wrapper for two stream split
 * 
 * @connections
 * @connect{pin[0], stream H*INP_W*sizeof(TT)}
 * @connect{pout[0], stream H*OUT_W*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitFilterStreamGraph : public adf::graph {

  public:
		static constexpr int LCNT = (INP_W - OUT_W) / (OUT_W - OVERLAP) + 1;
    adf::kernel k[1];
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    SplitFilterStreamGraph(
      int lane_idx
    ) { 
      k[0] = adf::kernel::create_object<SPLIT<TT, H, INP_W, OUT_W, OVERLAP>>(lane_idx);
      adf::source(k[0]) = "split.cc";
      adf::headers(k[0]) = {"split.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      
      adf::samples_per_iteration(k[0].in[0]) = H*INP_W;
      adf::samples_per_iteration(k[0].out[0]) = H*OUT_W;
    }

};


/**
 * @brief Graph wrapper for two stream split
 * 
 * @connections
 * @connect{pin[0], stream H*INP_W*sizeof(TT)}
 * @connect{pout[0], stream H*OUT_W*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitFilterStreamTwiceGraph : public adf::graph {

  public:
		static constexpr int LCNT = (INP_W - OUT_W) / (OUT_W - OVERLAP) + 1;
    adf::kernel k[1];
    adf::port<input> pin[1];
    adf::port<output> pout[2];

    SplitFilterStreamTwiceGraph(
      int lane_idx
    ) { 
      k[0] = adf::kernel::create_object<SPLIT<TT, H, INP_W, OUT_W, OVERLAP>>(lane_idx);
      adf::source(k[0]) = "split.cc";
      adf::headers(k[0]) = {"split.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::connect<adf::stream> (k[0].out[1], pout[1]);
      
      adf::samples_per_iteration(k[0].in[0]) = H*INP_W;
      adf::samples_per_iteration(k[0].out[0]) = H*OUT_W;
      adf::samples_per_iteration(k[0].out[1]) = H*OUT_W;
    }

};


/**
 * @brief Graph wrapper for two stream split
 * 
 * @connections
 * @connect{pin[0], stream H*INP_W*sizeof(TT)}
 * @connect{pout[0], stream H*OUT_W*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitFilterPktStreamGraph : public adf::graph {

  public:
		static constexpr int LCNT = (INP_W - OUT_W) / (OUT_W - OVERLAP) + 1;
    adf::kernel k[1];
    adf::port<input> pin[1];
    adf::port<output> pout[LCNT];

    adf::pktsplit<(LCNT+1)/2> sp0;
    adf::pktsplit<LCNT/2> sp1;

    SplitFilterPktStreamGraph() { 
      k[0] = adf::kernel::create_object<SPLIT<TT, H, INP_W, OUT_W, OVERLAP>>();
      adf::source(k[0]) = "split.cc";
      adf::headers(k[0]) = {"split.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::samples_per_iteration(k[0].in[0]) = H*INP_W;

      sp0 = adf::pktsplit<(LCNT+1)/2>::create();
      for (int i = 0; i < (LCNT+1)/2; i++) {
        adf::connect<adf::pktstream, adf::stream> (sp0.out[i], pout[2*i]);
        adf::samples_per_iteration(sp0.out[i]) = H*OUT_W;
      }
      adf::connect<adf::pktstream> (k[0].out[0], sp0.in[0]);

      sp1 = adf::pktsplit<LCNT/2>::create();
      for (int i = 0; i < LCNT/2; i++) {
        adf::connect<adf::pktstream, adf::stream> (sp1.out[i], pout[2*i+1]);
        adf::samples_per_iteration(sp1.out[i]) = H*OUT_W;
      }
      adf::connect<adf::pktstream> (k[0].out[1], sp1.in[0]);
    }

};
/** @} */


#endif // __SPLIT_GRAPH_H__