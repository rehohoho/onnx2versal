#ifndef __QGEMM_GRAPH_H_
#define __QGEMM_GRAPH_H_

#include <adf.h>
#include "qgemm.h"


/**
 * @defgroup Qgemm
 * 
 * @brief The qgemm operator consumes a quantized input tensor, its scale and zero point, 
 * a quantized weight, its scale and zero point, and output's scale and zero point, 
 * and computes the quantized output. xA^T + b as per torch.nn.Linear. 
 * Applies general matrix multiply: output(MxN) = input(MxK) * weights(KxNPAD) + bias(NPAD)
 * 
 * @tparam GEMM     Gemm Kernel
 * @tparam M        number of rows of input matrix
 * @tparam K        number of cols / number of rows of weight matrix
 * @tparam N        number of cols of weight matrix / size of bias vector
 * @tparam NPAD     padded N
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * 
 * @connections
 * @connect{pin[0], M*K*4}
 * @connect{pout[0], M*NPAD*4}
 * @endconnections
 */
template <template<int, int, int, int> class QGEMM, int M, int K, int N, int NPAD>
class QGemmGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QGemmGraph(
      std::vector<int8_t> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero_point,
      int8_t w_zero_point,
      int8_t y_zero_point
    ) { 
      k[0] = adf::kernel::create_object<QGEMM<M, K, N, NPAD>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero_point, w_zero_point, y_zero_point);
      adf::source(k[0]) = "qgemm.cc";
      adf::headers(k[0]) = {"qgemm.h"};
      adf::runtime<ratio>(k[0]) = 0.6;

      adf::connect<adf::window<M*K>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<M*NPAD>> (k[0].out[0], pout[0]);
    }

};
/** @} */


#endif // __QGEMM_GRAPH_H_