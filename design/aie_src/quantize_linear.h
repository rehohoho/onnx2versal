#ifndef QUANTIZE_LINEAR_H
#define QUANTIZE_LINEAR_H

#include <adf.h>
#include <assert.h>


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
 * @brief Scalar implementation, QuantizeLinearScalar<1*1*28*28> takes 92401 cycles
 */
template <int INP_H, int INP_W, int OUT_W>
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


/**
 * @brief Vector implementation, QuantizeLinearVector<1*1*28*28> takes 2457 cycles,
 * requires INP_W%4==0, OUT_W%16==0
 */
template <int INP_H, int INP_W, int OUT_W>
class QuantizeLinearVector {
  
  private:
    float y_scale;
    int y_zero_point; // same type as output

    // precompute
    int bitshift = 16;

    unsigned int select_mask;
    int16_t y_scale_inv_int;
    int32_t shift;
	
  public:
    QuantizeLinearVector (
      float y_scale,
      int y_zero_point
    );

		void filter(
			input_window<float>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%4 == 0 && OUT_W%16 == 0);
			REGISTER_FUNCTION(QuantizeLinearVector::filter);
		}
};
/** @}*/


#endif // QUANTIZE_LINEAR_H
