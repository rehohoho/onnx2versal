#ifndef __QLINEARPOOL_GRAPH_H__
#define __QLINEARPOOL_GRAPH_H__

#include <adf.h>
#include "qlinearpool.h"
#include "graph_concat.h"
#include "graph_split.h"


/**
 * @defgroup QLinearPool2D
 * 
 * @brief QLinearPool2D function on BCHW, yielding BCH'W', where H'=H/factor, W'=W/factor
 * Scalar kernels allow W'<W/factor.
 * 
 * @tparam TT           input and output type
 * @tparam QLINEARPOOL  QLinearPool Kernel
 * @tparam INP_H        input height, used to calculate pool factor
 * @tparam INP_W        input width
 * @tparam OUT_W        output height, used to calculate pool factor
 * @tparam OUT_W        output width
 * @tparam B            batch size
 * @tparam C            input channels
 * @tparam KH           kernel height
 * @tparam KW           kernel width
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W*C*sizeof(TT)}
 * @connect{pout[0], stream B*OUT_H*OUT_W*C*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int, int, int> class QLINEARPOOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class QLinearPoolGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QLinearPoolGraph(
      float in_scale,
      float out_scale,
      int8_t in_zero,
      int8_t out_zero
    ) { 
      k[0] = adf::kernel::create_object<QLINEARPOOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW>>(
        in_scale, out_scale, in_zero, out_zero);
      adf::source(k[0]) = "qlinearpool.cc";
      adf::headers(k[0]) = {"qlinearpool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_H*INP_W*C*sizeof(TT)>> (pin[0], k[0].in[0]);
      adf::connect<adf::stream>                             (k[0].out[0], pout[0]);

      adf::samples_per_iteration(k[0].out[0]) = B*C*OUT_H*OUT_W;
    }

};


/**
 * @brief Multi instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W*C*sizeof(TT)}
 * @connect{pout[0], stream B*OUT_H*OUT_W*C*sizeof(TT)}
 * @endconnections
 */
template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int> class QLINEARPOOL,
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK, 
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class QLinearPoolChunkCGraph : public adf::graph {

  private:
    static constexpr int LCNT = C/CCHUNK;
    ConcatStreamGraph<CONCAT, TT, LCNT, B, CCHUNK*OUT_H*OUT_W, C*OUT_H*OUT_W> concat_graph;

    adf::kernel split[(LCNT+1)/2];
    adf::kernel k[LCNT];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QLinearPoolChunkCGraph(
      float in_scale,
      float out_scale,
      int8_t in_zero,
      int8_t out_zero
    ) { 
      static_assert(C % CCHUNK == 0);
      static_assert(LCNT <= 8);

      for (int i = 0; i < LCNT/2; i++) {
        split[i] = adf::kernel::create_object<SplitFilterInt8StreamTwice<TT, B, C*INP_H*INP_W, CCHUNK*INP_H*INP_W, 0>>(i*2);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::connect<adf::stream> (pin[0], split[i].in[0]);
        adf::samples_per_iteration(split[i].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(split[i].out[0]) = B*CCHUNK*INP_H*INP_W;
        adf::samples_per_iteration(split[i].out[1]) = B*CCHUNK*INP_H*INP_W;
      }
      if ((LCNT & 0x1) == 1) {
        int i = (LCNT+1)/2 - 1;
        split[i] = adf::kernel::create_object<SplitFilterInt8Stream<TT, B, C*INP_H*INP_W, CCHUNK*INP_H*INP_W, 0>>(LCNT-1);
        adf::source(split[i]) = "split.cc";
        adf::headers(split[i]) = {"split.h"};
        adf::runtime<ratio>(split[i]) = 0.6;

        adf::connect<adf::stream> (pin[0], split[i].in[0]);
        adf::samples_per_iteration(split[i].in[0]) = B*C*INP_H*INP_W;
        adf::samples_per_iteration(split[i].out[0]) = B*CCHUNK*INP_H*INP_W;
      }

      for (int i = 0; i < LCNT; i++) {
        k[i] = adf::kernel::create_object<QLINEARPOOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, CCHUNK, KH, KW>>(
          in_scale, out_scale, in_zero, out_zero);
        adf::source(k[i]) = "qlinearpool.cc";
        adf::headers(k[i]) = {"qlinearpool.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        
        adf::connect<adf::window<B*CCHUNK*INP_H*INP_W>> (split[i/2].out[i&0x1], k[i].in[0]);
        adf::connect<adf::stream> (k[i].out[0], concat_graph.pin[i]);
        adf::samples_per_iteration(k[i].out[0]) = B*CCHUNK*OUT_H*OUT_W;
      }
      
      adf::connect<adf::stream> (concat_graph.pout[0], pout[0]);
    }

};


/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], stream B*INP_H*INP_W*C*sizeof(TT)}
 * @connect{pout[0], stream B*OUT_H*OUT_W*C*sizeof(TT)}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int, int, int> class QLINEARPOOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class QLinearPoolStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QLinearPoolStreamGraph(
      float in_scale,
      float out_scale,
      int8_t in_zero,
      int8_t out_zero
    ) { 
      k[0] = adf::kernel::create_object<QLINEARPOOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW>>(
        in_scale, out_scale, in_zero, out_zero);
      adf::source(k[0]) = "qlinearpool.cc";
      adf::headers(k[0]) = {"qlinearpool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);

      adf::samples_per_iteration(k[0].out[0]) = B*C*OUT_H*OUT_W;
    }

};
/** @} */


#endif // __QLINEARPOOL_GRAPH_H__