#ifndef __POOL_GRAPH_H__
#define __POOL_GRAPH_H__

#include <adf.h>
#include "pool.h"
#include "graph_split.h"
#include "graph_concat.h"


/**
 * @defgroup Pool2D
 * 
 * @brief Pool2D function on BCHW, yielding BCH'W', where H'=H/factor, W'=W/factor
 * Scalar kernels allow W'<W/factor.
 * 
 * @tparam TT         input and output type
 * @tparam POOL       Pool Kernel
 * @tparam INP_H      input height, used to calculate pool factor
 * @tparam INP_W      input width, used to calculate pool factor
 * @tparam OUT_H      output height, used to calculate pool factor
 * @tparam OUT_W      output width, used to calculate pool factor
 * @tparam B          batch size
 * @tparam C          input channels
 * @tparam KH         kernel height
 * @tparam KW         kernel width
 * @tparam STEP_H     stride in the height dimension
 * @tparam STEP_W     stride in the width dimension
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W*C*sizeof(TT)}
 * @connect{pout[0], B*OUT_H*OUT_W*C*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int, int, int, int, int> class POOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class PoolGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    PoolGraph() { 
      k[0] = adf::kernel::create_object<POOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>>();
      adf::source(k[0]) = "pool.cc";
      adf::headers(k[0]) = {"pool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_H*INP_W*C*sizeof(TT)>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_H*OUT_W*C*sizeof(TT)>> (k[0].out[0], pout[0]);
    }

};


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int, int, int> class POOL,
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
class PoolChunkCGraph : public adf::graph {

  private:
    typedef SplitGraph<SPLIT, TT, B, C*INP_H*INP_W, CCHUNK*INP_H*INP_W, 0> mSplitGraph;
    mSplitGraph split_graph;
    static constexpr int LCNT = mSplitGraph::LCNT;
    ConcatStreamGraph<CONCAT, TT, LCNT, B, CCHUNK*OUT_H*OUT_W, C*OUT_H*OUT_W> concat_graph;

    adf::kernel k[LCNT];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    PoolChunkCGraph() { 
      static_assert(LCNT <= 8);
      adf::connect<adf::stream> (pin[0], split_graph.pin[0]);

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<POOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, CCHUNK, KH, KW, STEP_H, STEP_W>>();
        adf::source(k[i]) = "pool.cc";
        adf::headers(k[i]) = {"pool.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        
        adf::connect<adf::window<B*CCHUNK*INP_H*INP_W*sizeof(TT)>> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::window<B*CCHUNK*OUT_H*OUT_W*sizeof(TT)>> (k[i].out[0], concat_graph.pin[i]);
      }

      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);
    }

};
/** @} */


#endif // __POOL_GRAPH_H__