#ifndef __TRANSPOSE_GRAPH_H__
#define __TRANSPOSE_GRAPH_H__

#include <adf.h>
#include "transpose.h"
#include "split.h"
#include "graph_conv.h"
#include "graph_utils.h"


/**
 * @defgroup Transpose
 * 
 * @brief Transpose function, BHWC->BCHW tentatively
 * 
 * @tparam TRANSPOSE  Transpose Kernel
 * @tparam B          batch size
 * @tparam H          height
 * @tparam W          width
 * @tparam C          input channels
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*H*W*C*sizeof(TT)}
 * @connect{pout[0], B*H*W*C*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int> class TRANSPOSE, 
  typename TT, int B, int H, int W, int C>
class TransposeGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    TransposeGraph() { 
      k[0] = adf::kernel::create_object<TRANSPOSE<TT, B, H, W, C>>();
      adf::source(k[0]) = "transpose.cc";
      adf::headers(k[0]) = {"transpose.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*H*W*C*sizeof(TT)>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*C*H*W*sizeof(TT)>> (k[0].out[0], pout[0]);
    }

};


/**
 * @brief Multi instance graph
 * 
 * @connections
 * @connect{pin[0], B*H*W*C*sizeof(TT)}
 * @connect{pout[0], B*H*W*C*sizeof(TT)}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int> class TRANSPOSE, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int B, int H, int W, int C>
class TransposeChunkHPktStreamGraph : public adf::graph {

  private:
    static constexpr int LCNT = H / HCHUNK;
    typedef SplitFilterPktStreamGraph<SPLIT, TT, B, H*W*C, HCHUNK*W*C, 0> mSplitGraph;
    mSplitGraph split_graph;

    adf::kernel k[LCNT];
    ConcatStreamGraph<CONCAT, TT, LCNT, B*C, HCHUNK*W, H*W> concat_graph;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    TransposeChunkHPktStreamGraph() { 
      static_assert(H % HCHUNK == 0);
      static_assert(LCNT <= 8);
      static_assert(B*HCHUNK*W*C*4 <= TILE_BYTES);

      adf::connect<adf::stream> (pin[0], split_graph.pin[0]);

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<TRANSPOSE<TT, B, HCHUNK, W, C>>();
        adf::source(k[i]) = "transpose.cc";
        adf::headers(k[i]) = {"transpose.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::single_buffer(k[i].in[0]);
        adf::single_buffer(k[i].out[0]);
        
        adf::connect<adf::pktstream, adf::window<B*HCHUNK*W*C*sizeof(TT)>> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::window<B*C*HCHUNK*W*sizeof(TT)>> (k[i].out[0], concat_graph.pin[i]);
      }
      
      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);
      
    }

};
/** @} */


#endif // __TRANSPOSE_GRAPH_H__