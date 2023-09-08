#ifndef __POOL_GRAPH_H__
#define __POOL_GRAPH_H__

#include <adf.h>
#include "pad.h"
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
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W,
  template<typename, int, int, int, int, int, int, int, int> class PAD, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class PoolGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    adf::kernel k[1];
    std::string id;
    std::vector<adf::kernel> pad;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    PoolGraph() { 
      k[0] = adf::kernel::create_object<POOL<TT, PAD_H, PAD_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>>();
      adf::source(k[0]) = "pool.cc";
      adf::headers(k[0]) = {"pool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*OUT_H*OUT_W*C*sizeof(TT)>> (k[0].out[0], pout[0]);
      
      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<PAD<TT, B*C, INP_H, INP_W, INP_W, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::window<B*PAD_H*PAD_W*C*sizeof(TT)>> (pad[0].out[0], k[0].in[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        adf::connect<adf::window<B*INP_H*INP_W*C*sizeof(TT)>> (pin[0], k[0].in[0]);
      }
    }

};


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int, int, int> class POOL,
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W,
  template<typename, int, int, int, int, int, int, int, int> class PAD, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class PoolChunkCGraph : public adf::graph {

  private:
    static constexpr int PAD_H = INP_H + H0 + H1;
    static constexpr int PAD_W = INP_W + W0 + W1;

    typedef SplitFilterPktStreamGraph<SPLIT, TT, B, C*PAD_H*PAD_W, CCHUNK*PAD_H*PAD_W, 0> mSplitGraph;
    mSplitGraph split_graph;
    static constexpr int LCNT = mSplitGraph::LCNT;
    ConcatStreamGraph<CONCAT, TT, LCNT, B, CCHUNK*OUT_H*OUT_W, C*OUT_H*OUT_W> concat_graph;

    std::vector<adf::kernel> pad;
    adf::kernel k[LCNT];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    PoolChunkCGraph(
      int repeat_cnt = 1
    ) { 
      static_assert(LCNT <= 8);

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<POOL<TT, PAD_H, PAD_W, OUT_H, OUT_W, B, CCHUNK, KH, KW, STEP_H, STEP_W>>();
        adf::source(k[i]) = "pool.cc";
        adf::headers(k[i]) = {"pool.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::repetition_count(k[i]) = repeat_cnt;
        
        adf::connect<adf::window<B*CCHUNK*PAD_H*PAD_W*sizeof(TT)>> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::window<B*CCHUNK*OUT_H*OUT_W*sizeof(TT)>> (k[i].out[0], concat_graph.pin[i]);
      }

      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);

      if (H0+H1+W0+W1 != 0) {
        pad.push_back(
          adf::kernel::create_object<PAD<TT, B*C, INP_H, INP_W, INP_W, H0, H1, W0, W1>>());
        adf::source(pad[0]) = "pad.cc";
        adf::headers(pad[0]) = {"pad.h"};
        adf::runtime<ratio>(pad[0]) = 0.6;
        adf::repetition_count(pad[0]) = repeat_cnt;

        adf::connect<adf::stream> (pin[0], pad[0].in[0]);
        adf::connect<adf::stream> (pad[0].out[0], split_graph.pin[0]);
        
        adf::samples_per_iteration(pad[0].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(pad[0].out[0]) = B*C*PAD_H*PAD_W;
      } else {
        adf::connect<adf::stream> (pin[0], split_graph.pin[0]);
      }

      // location constraints
      for (int i = 0; i < LCNT; i++) {
        if ((i&0x3) == 1) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=1, .row_offset=1});
        } else if ((i&0x3) == 3) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=-1, .row_offset=1});
        } else if ((i&0x3) == 2) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=1, .row_offset=0});
        } else if (i != 0) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=-1, .row_offset=0});
        }
        adf::location<adf::buffer>(k[i].in[0]) = adf::location<adf::kernel>(k[i]);
      }

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(k[2*i]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        
        adf::location_constraint cTilePos = adf::location<adf::kernel>(concat_graph.k1[i]);
        adf::location<adf::buffer>(k[i*2].out[0]) = cTilePos;
        adf::location<adf::buffer>(k[i*2].out[0]) = {adf::offset(0), adf::offset(8192)};
        adf::location<adf::stack>(k[i*2]) = cTilePos;
        adf::location<adf::buffer>(k[i*2+1].out[0]) = cTilePos;
        adf::location<adf::buffer>(k[i*2+1].out[0]) = {adf::offset(16384), adf::offset(24576)};
        adf::location<adf::stack>(k[i*2+1]) = cTilePos;
        adf::location<adf::stack>(concat_graph.k1[i]) = cTilePos;
      }
    }

};
/** @} */


#endif // __POOL_GRAPH_H__