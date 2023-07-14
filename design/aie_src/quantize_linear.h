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
 * @brief Scalar implementation, 
 * requires INP_W <= OUT_W, 
 * QuantizeLinearScalar<1*1*28*28> takes 92121 cycles
 */
template <typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearScalar {
  
  private:
    float y_scale;
    TT y_zero; // same type as output
	
  public:
    QuantizeLinearScalar (
      float y_scale,
      TT y_zero
    ): y_scale(y_scale), y_zero(y_zero) {};

		void filter(
			input_window<float>* in,
			output_window<TT>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W <= OUT_W);
			REGISTER_FUNCTION(QuantizeLinearScalar::filter);
		}
};


/**
 * @brief Vector implementation, 
 * requires INP_W%4==0, OUT_W%16==0, INP_W <= OUT_W, 
 * QuantizeLinear<1*1*28*28> takes 1977 cycles
 */
template <typename TT, int INP_H, int INP_W, int OUT_W>
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
      static_assert(INP_W <= OUT_W);
			REGISTER_FUNCTION(QuantizeLinear::filter);
		}
};


/**
 * @brief Vector implementation,
 * requires INP_W%4==0, OUT_W%16==0, INP_W <= OUT_W, 
 * QuantizeLinearFmul<1*1*28*28> takes 1534 cycles,
 */
template <typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearFmul {
  
  private:
    float y_scale;
    TT y_zero; // same type as output

  public:
    QuantizeLinearFmul (
      float y_scale,
      TT y_zero
    ): y_scale(y_scale), y_zero(y_zero) {};

		void filter(
			input_window<float>* in,
			output_window<TT>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%4 == 0 && OUT_W%16 == 0);
      static_assert(INP_W <= OUT_W);
			REGISTER_FUNCTION(QuantizeLinearFmul::filter);
		}
};


/**
 * @brief Vector stream implementation,
 * requires INP_W%4==0, OUT_W%16==0, INP_W <= OUT_W, 
 * QuantizeLinearFmulStream<1*1*28*28> takes 1750 cycles,
 */
template <typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearFmulStream {
  
  private:
    float y_scale;
    TT y_zero; // same type as output

  public:
    QuantizeLinearFmulStream (
      float y_scale,
      TT y_zero
    ): y_scale(y_scale), y_zero(y_zero) {};

		void filter(
			input_stream<float>* in,
			output_stream<TT>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%4 == 0 && OUT_W%16 == 0);
      static_assert(INP_W <= OUT_W);
			REGISTER_FUNCTION(QuantizeLinearFmulStream::filter);
		}
};
/** @}*/


#endif // QUANTIZE_LINEAR_H
