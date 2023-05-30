#ifndef QLINEARCONV_H_
#define QLINEARCONV_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup QLinearConvKernels
 * @ingroup QLinearConv
 * 
 * 
 * 
 * @details See https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv.
 * - y = saturate ((x / y_scale) + y_zero_point)
 * - Bias must be quantized using scale = x_scale * w_scale and zero_point = 0
 * 
 * Computation
 * - x = (qx - qx_zero) * qx_scale
 * - bias = qbias * x_scale * w_scale
 * - y = x*w + bias =>
 * - (qy-qy_zero)*qy_scale = (qx-qx_zero)*qx_scale * (qw-qw_zero)*qw_scale + qbias*qx_scale*qw_scale
 *                       = [(qx-qx_zero) * (qw-qw_zero) + qbias] * qx_scale*qw_scale
 * - qy = qy_zero + [(qx-qx_zero)*(qw-qw_zero) + qbias] * qx_scale*qw_scale/qy_scale
 * 
 * Implementation
 * - only precompute -qx_zero*(qw_qw_zero), rounding is done before adding qy_zero
 * - int32 bias: -qx_zero*(qw_zero): k*int8*int8 > 16bits
 * - int8 shifted qy_zero: shift added into acc
 * - int16 scale: saturated to 8 bits
 */


/**
 * @brief Scalar implementation, QLinearConvScalar<28,24,1,1,6,5> takes 1027692 cycles
 */
template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
class QLinearConvScalar {
  
  private:
    alignas(32) int8_t (&weights)[M*C*K*16];
    alignas(32) int32_t (&bias)[M];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero_point;
    int8_t w_zero_point;
    int8_t y_zero_point;

    float scale;
	
  public:
    QLinearConvScalar (
      int8_t (&w)[M*C*K*16],
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero_point,
      int8_t w_zero_point,
      int8_t y_zero_point
    ): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero_point(x_zero_point), w_zero_point(w_zero_point), y_zero_point(y_zero_point) {
      scale = x_scale*w_scale/y_scale;
    };

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


/**
 * @brief Vector implementation, QLinearConvVector<28,24,1,1,6,5> takes 3237 cycles.
 * Requires data to be arranged in [a,b,c,d,e] -> [0,0,0,0,a,a,b,b,c,c,d,d,e,e,0,0], 
 * due to int8 indexing restriction. Requires INP_W%16=0, OUT_W%16=0
 */
template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
class QLinearConvVector {
  
  private:
    alignas(32) int8_t (&weights)[M*C*K*16];
    alignas(32) int32_t (&bias)[M];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero_point;
    int8_t w_zero_point;
    int8_t y_zero_point;

    // precomputation
    int scalebits;
    int16_t scale;
	
  public:
    QLinearConvVector (
      int8_t (&w)[M*C*K*16],
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero_point,
      int8_t w_zero_point,
      int8_t y_zero_point
    );

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%16==0 && OUT_W%16==0);
			REGISTER_FUNCTION(QLinearConvVector::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // QLINEARCONV_H_
