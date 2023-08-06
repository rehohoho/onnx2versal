#ifndef __QUANTIZE_LINEAR_GRAPH_H__
#define __QUANTIZE_LINEAR_GRAPH_H__

#include <adf.h>
#include "quantize_linear.h"
#include "graph_concat.h"
#include "graph_split.h"


/**
 * @defgroup QuantizeLinear
 * 
 * @brief Linear quantization operator. It consumes a high precision tensor, a scale, and 
 * a zero point to compute the low precision / quantized tensor. The quantization formula 
 * is y = saturate ((x / y_scale) + y_zero). For saturation, it saturates to [0, 255] 
 * if it's uint8, or [-128, 127] if it's int8. For (x / y_scale), it's rounding to the 
 * nearest even. 
 * 
 * @tparam QUANTIZE_LINAER  QuantizeLinaer Kernel
 * @tparam INP_H	          input height
 * @tparam INP_W	          input width
 * @tparam OUT_W	          output width, allows padding, expects OUT_W >= INP_W
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], INP_H*INP_W*4}
 * @connect{pout[0], INP_H*OUT_W}
 * @endconnections
 */
template <template<typename, int, int, int> class QUANTIZE_LINEAR, typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QuantizeLinearGraph(
      float y_scale,
      TT y_zero,
      int repeat_cnt = 1
    ) { 
      k[0] = adf::kernel::create_object<QUANTIZE_LINEAR<TT, INP_H, INP_W, OUT_W>>(y_scale, y_zero);
      adf::source(k[0]) = "quantize_linear.cc";
      adf::headers(k[0]) = {"quantize_linear.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::repetition_count(k[0]) = repeat_cnt;
      
      adf::connect<adf::window<INP_H*INP_W*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<INP_H*OUT_W>> (k[0].out[0], pout[0]);
    }

};


/**
 * @brief Single instance stream graph
 * 
 * @connections
 * @connect{pin[0], stream INP_H*INP_W*4 bytes}
 * @connect{pout[0], stream INP_H*OUT_W bytes}
 * @endconnections
 */
template <template<typename, int, int, int> class QUANTIZE_LINEAR, typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QuantizeLinearStreamGraph(
      float y_scale,
      TT y_zero
    ) { 
      k[0] = adf::kernel::create_object<QUANTIZE_LINEAR<TT, INP_H, INP_W, OUT_W>>(y_scale, y_zero);
      adf::source(k[0]) = "quantize_linear.cc";
      adf::headers(k[0]) = {"quantize_linear.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);

      adf::samples_per_iteration(k[0].in[0]) = INP_H*INP_W;
      adf::samples_per_iteration(k[0].out[0]) = INP_H*OUT_W;
    }

};


/**
 * @brief Multi instance pktstream graph
 * 
 * @connections
 * @connect{pin[0], stream INP_H*INP_W*4 bytes}
 * @connect{pout[0], stream INP_H*OUT_W bytes}
 * @endconnections
 */
template <template<typename, int, int, int> class QUANTIZE_LINEAR, int HCHUNK,
  typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearChunkHPktStreamGraph : public adf::graph {

  private:
    typedef SplitFilterPktStreamGraph<SplitFilterFloatPktStream, float, 1, INP_H*INP_W, HCHUNK*INP_W, 0> mSplitGraph;
    mSplitGraph split_graph;

    static constexpr int LCNT = INP_H/HCHUNK;
    adf::kernel k[LCNT];
    std::string id;

    ConcatStreamGraph<ConcatInt8Stream, TT, LCNT, 1, HCHUNK*OUT_W, INP_H*OUT_W> concat_graph;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QuantizeLinearChunkHPktStreamGraph(
      float y_scale,
      TT y_zero
    ) { 
      static_assert(INP_H % HCHUNK == 0);

      adf::connect<adf::stream> (pin[0], split_graph.pin[0]);

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<QUANTIZE_LINEAR<TT, HCHUNK, INP_W, OUT_W>>(y_scale, y_zero);
        adf::source(k[i]) = "quantize_linear.cc";
        adf::headers(k[i]) = {"quantize_linear.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        
        adf::connect<adf::stream> (split_graph.pout[i], k[i].in[0]);
        adf::connect<adf::stream> (k[i].out[0], concat_graph.pin[i]);

        adf::samples_per_iteration(k[i].in[0]) = HCHUNK*INP_W;
        adf::samples_per_iteration(k[i].out[0]) = HCHUNK*OUT_W;

        if ((i&0x1) == 1) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        }
        if (i == 2) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=0, .row_offset=2});
        }
        adf::location<adf::stack>(k[i]) = adf::location<adf::kernel>(k[i]);
      }

      for (int i = 0; i < concat_graph.k1.size(); i++) {
        adf::location<adf::kernel>(concat_graph.k1[i]) = 
          adf::location<adf::kernel>(k[i*2+1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        adf::location<adf::stack>(concat_graph.k1[i]) = adf::location<adf::kernel>(concat_graph.k1[i]);
      }

      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);
    }

};
/** @} */


#endif // __QUANTIZE_LINEAR_GRAPH_H__