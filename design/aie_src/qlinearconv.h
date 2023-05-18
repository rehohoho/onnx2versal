#ifndef QLINAERCONV_H_
#define QLINAERCONV_H_

#include <adf.h>


/** 
 * @defgroup QLinearConvKernels
 * @ingroup QLinearConv
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv
 * Note:
 *  y = saturate ((x / y_scale) + y_zero_point)
 *  x = (y - y_zero_point) * y_scale
 * 
 *  x3 = x1*x2
 *  (y3-y3_zero_point)*y3_scale = (y1-y1_zero_point)*y1_scale * (y2-y2_zero_point)*y2_scale
 *  y3 = y3_zero_point + (y1-y1_zero_point)*y1_scale * (y2-y2_zero_point)*y2_scale / y3_scale
 *     = y3_zero_point + y1_scale*y2_scale/y3_scale * (y1-y1_zero_point) * (y2-y2_zero_point)
 * 
 * @{
 */


/**
 * @brief Scalar implementation, QLinearConvScalar<28,24,1,1,6,5> takes  cycles
 */
template <int INP_W, int OUT_W, int B, int C, int M, int K>
class QLinearConvScalar {
  
  private:
    alignas(32) int8_t (&weights)[M*C*K*K];
    alignas(32) int8_t (&bias)[M];
    float xy_over_w;
    int8_t x_zero_point;
    int8_t y_zero_point;
	
  public:
    QLinearConvScalar (
      int8_t (&w)[M*C*K*K],
      int8_t (&b)[M],
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


#endif // QLINAERCONV_H_
