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
  template<typename, int, int, int, int> class TRANSPOSE, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int B, int H, int W, int C>
class TransposeHChunkGraph : public adf::graph {

  private:
    static constexpr int LCNT = H / HCHUNK;
    adf::kernel split[(LCNT+1)/2];
    adf::kernel k[LCNT];
    ConcatStreamGraph<CONCAT, float_t, LCNT, B*C, HCHUNK*W, H*W> concat_graph;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    TransposeHChunkGraph() { 
      static_assert(H % HCHUNK == 0);
      static_assert(LCNT <= 8);
      static_assert(B*HCHUNK*W*C*4 <= TILE_BYTES);

      for (int i = 0; i < LCNT/2; i++) {
        split[i] = adf::kernel::create_object<SplitFilterFloatStreamTwice<float_t, B, H*W*C, HCHUNK*W*C, 0>>(i*2);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::connect<adf::stream> (pin[0], split[i].in[0]);

        adf::samples_per_iteration(split[i].in[0]) = B*H*W*C;
        adf::samples_per_iteration(split[i].out[0]) = B*HCHUNK*W*C;
        adf::samples_per_iteration(split[i].out[1]) = B*HCHUNK*W*C;
      }
      if ((LCNT & 0x1) == 1) {
        int i = (LCNT+1)/2 - 1;
        split[i] = adf::kernel::create_object<SplitFilterFloatStream<float_t, B, H*W*C, HCHUNK*W*C, 0>>(LCNT-1);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;
        
        adf::connect<adf::stream> (pin[0], split[i].in[0]);

        adf::samples_per_iteration(split[i].in[0]) = B*H*W*C;
        adf::samples_per_iteration(split[i].out[0]) = B*HCHUNK*W*C;
      }

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<TRANSPOSE<TT, B, HCHUNK, W, C>>();
        adf::source(k[i]) = "transpose.cc";
        adf::headers(k[i]) = {"transpose.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::single_buffer(k[i].in[0]);
        adf::single_buffer(k[i].out[0]);
        
        adf::connect<adf::window<B*HCHUNK*W*C*sizeof(TT)>> (split[i/2].out[i&0x1], k[i].in[0]);
        adf::connect<adf::window<B*C*HCHUNK*W*sizeof(TT)>> (k[i].out[0], concat_graph.pin[i]);
        
        if (i >= 1) {
          adf::location<adf::kernel>(k[i]) =
            adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=1, .row_offset=0});
        }
        adf::location_constraint kTilePos = adf::location<adf::kernel>(k[i]);
        adf::location<adf::buffer>(k[i].out[0]) = kTilePos;
        adf::location<adf::buffer>(k[i].out[0]) = {adf::offset(0)};
        
        if (i % 2 == 1) {
          adf::location<adf::kernel>(split[i/2]) = 
            adf::location<adf::kernel>(k[i]) + adf::relative_offset({.col_offset=0, .row_offset=1});
          adf::location<adf::buffer>(k[i].in[0]) = adf::location<adf::kernel>(split[i/2]);
          adf::location<adf::buffer>(k[i].in[0]) = {adf::offset(0)};
        }
      }
      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(k[i*2]) + adf::relative_offset({.col_offset=0, .row_offset=-1});
        
        adf::location_constraint cTilePos = adf::location<adf::kernel>(concat_graph.k1[i]);
        adf::location<adf::stack>(k[i*2]) = cTilePos;
        adf::location<adf::stack>(concat_graph.k1[i]) = cTilePos;
      }
      
    }

};
/** @} */


#endif // __TRANSPOSE_GRAPH_H__