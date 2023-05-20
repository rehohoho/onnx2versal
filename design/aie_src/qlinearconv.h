#ifndef QLINEARCONV_H_
#define QLINEARCONV_H_

#include <adf.h>


/** 
 * @defgroup QLinearConvKernels
 * @ingroup QLinearConv
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv
 * Note:
 *  y = saturate ((x / y_scale) + y_zero_point)
 *  Bias must be quantized using scale = x_scale * w_scale and zero_point = 0
 *  Saturate at the end only.
 * 
 * Computation:
 *  x = (qx - qx_zero) * qx_scale
 *  bias = qbias * x_scale * w_scale
 *  y = x*w + bias =>
 *  (qy-qy_zero)*qy_scale = (qx-qx_zero)*qx_scale * (qw-qw_zero)*qw_scale + qbias*qx_scale*qw_scale
 *                       = [(qx-qx_zero) * (qw-qw_zero) + qbias] * qx_scale*qw_scale
 *  qy = qy_zero + [(qx-qx_zero)*(qw-qw_zero) + qbias] * qx_scale*qw_scale/qy_scale
 * 
 */


/**
 * @brief Scalar implementation, QLinearConvScalar<28,24,1,1,6,5> takes  cycles
 */
template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
class QLinearConvScalar {
  
  private:
    alignas(32) int8_t (&weights)[M*C*K*K];
    alignas(32) int32_t (&bias)[M];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero_point;
    int8_t w_zero_point;
    int8_t y_zero_point;
	
  public:
    QLinearConvScalar (
      int8_t (&w)[M*C*K*K],
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero_point,
      int8_t w_zero_point,
      int8_t y_zero_point
    ): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero_point(x_zero_point), w_zero_point(w_zero_point), y_zero_point(y_zero_point) {};

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QLinearConvScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // QLINEARCONV_H_