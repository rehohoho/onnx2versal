#ifndef DEQUANTIZE_LINEAR_H
#define DEQUANTIZE_LINEAR_H

#include <adf.h>


/** 
 * @defgroup DequantizeLinearKernels
 * @ingroup DequantizeLinear
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear
 * y = (x - x_zero_point) * x_scale
 * 
 * @{
 */


/**
 * @brief Scalar implementation,
 * DequantizeLinearScalar<96,84> takes 296 cycles
 */
template <int B, int INP_W, int OUT_W>
class DequantizeLinearScalar {
  
  private:
    float scale;
    int8_t zero; // same type as output
	
  public:
    DequantizeLinearScalar (
      float scale,
      int8_t zero
    ): scale(scale), zero(zero) {};

		void filter(
			input_window<int8_t>* in,
			output_window<float>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(DequantizeLinearScalar::filter);
		}
};
/** @}*/


#endif // DEQUANTIZE_LINEAR_H
