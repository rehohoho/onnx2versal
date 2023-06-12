#ifndef QUANTIZE_LINEAR_H
#define QUANTIZE_LINEAR_H

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup QuantizeLinearKernels
 * @ingroup QuantizeLinear
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
 * y = saturate ((x / y_scale) + y_zero)
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
    int y_zero; // same type as output
	
  public:
    QuantizeLinearScalar (
      float y_scale,
      int y_zero
    ): y_scale(y_scale), y_zero(y_zero) {};

		void filter(
			input_window<float>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QuantizeLinearScalar::filter);
		}
};


/**
 * @brief Vector implementation, QuantizeLinear<1*1*28*28> takes 1945 cycles,
 * requires INP_W%4==0, OUT_W%16==0
 */
template <int INP_H, int INP_W, int OUT_W>
class QuantizeLinear {
  
  private:
    float y_scale;
    int8_t y_zero; // same type as output

    // precompute
    int xbitshift = 16; // ybitshift in [0:16], acc48 result
    int ybitshift;
    int16_t y_scale_inv_int;
	
  public:
    QuantizeLinear (
      float y_scale,
      int8_t y_zero
    );

		void filter(
			input_window<float>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%4 == 0 && OUT_W%16 == 0);
			REGISTER_FUNCTION(QuantizeLinear::filter);
		}
};


/**
 * @brief Vector implementation, QuantizeLinearFmul<1*1*28*28> takes 1526 cycles,
 * requires INP_W%4==0, OUT_W%16==0
 */
template <int INP_H, int INP_W, int OUT_W>
class QuantizeLinearFmul {
  
  private:
    float y_scale;
    int8_t y_zero; // same type as output

  public:
    QuantizeLinearFmul (
      float y_scale,
      int8_t y_zero
    ): y_scale(y_scale), y_zero(y_zero) {};

		void filter(
			input_window<float>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%4 == 0 && OUT_W%16 == 0);
			REGISTER_FUNCTION(QuantizeLinearFmul::filter);
		}
};
/** @}*/


#endif // QUANTIZE_LINEAR_H
