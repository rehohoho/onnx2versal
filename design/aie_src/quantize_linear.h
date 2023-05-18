#ifndef QUANTIZE_LINEAR_H
#define QUANTIZE_LINEAR_H

#include <adf.h>


/** 
 * @defgroup QuantizeLinearKernels
 * @ingroup QuantizeLinear
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
 * y = saturate ((x / y_scale) + y_zero_point)
 * 
 * @{
 */


/**
 * @brief Scalar implementation, QuantizeLinearScalar<1*1*28*28> takes  cycles
 */
template <int WINDOW_SIZE>
class QuantizeLinearScalar {
  
  private:
    float y_scale;
    int y_zero_point; // same type as output
	
  public:
    QuantizeLinearScalar (
      float y_scale,
      int y_zero_point
    ): y_scale(y_scale), y_zero_point(y_zero_point) {};

		void filter(
			input_window<float>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QuantizeLinearScalar::filter);
		}
};
/** @}*/


#endif // QUANTIZE_LINEAR_H
