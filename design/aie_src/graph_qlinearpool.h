#ifndef __QLINEARPOOL_GRAPH_H__
#define __QLINEARPOOL_GRAPH_H__

#include <adf.h>
#include "qlinearpool.h"


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
 * 
 * @{
 */

/**
 * @brief Single instance graph
 * 
 * @connections
 * @connect{pin[0], B*INP_H*INP_W*C*TTSIZE}
 * @connect{pout[0], B*OUT_H*OUT_W*C*TTSIZE}
 * @endconnections
 */
template <template<typename, int, int, int, int, int, int> class QLINEARPOOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
class QLinearPoolGraph : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    static constexpr int K = INP_H / OUT_H;
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
      k[0] = adf::kernel::create_object<QLINEARPOOL<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>>(
        in_scale, out_scale, in_zero, out_zero);
      adf::source(k[0]) = "qlinearpool.cc";
      adf::headers(k[0]) = {"qlinearpool.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      
      adf::connect<adf::window<B*INP_H*INP_W*C*TTSIZE>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_H*OUT_W*C*TTSIZE>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __QLINEARPOOL_GRAPH_H__