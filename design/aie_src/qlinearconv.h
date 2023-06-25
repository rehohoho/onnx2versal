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
 * - y = saturate ((x / y_scale) + y_zero)
 * - Bias must be quantized using scale = x_scale * w_scale and zero = 0
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
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int K>
class QLinearConvScalar {
  
  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;

    alignas(32) int8_t (&weights)[M*C*K*16];
    alignas(32) int32_t (&bias)[M];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    float scale;
	
  public:
    QLinearConvScalar (
      int8_t (&w)[M*C*K*16],
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    ): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero(x_zero), w_zero(w_zero), y_zero(y_zero) {
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
 * @brief Scalar implementation streaming weights, QLinearConvScalar<28,24,1,1,6,5> takes 1027692 cycles,
 * expects weights stream to be padded from MxCxKxK to MxCx16, K < 5
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int K>
class QLinearConvScalarStream {
  
  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    static constexpr int CKK_ROW_SIZE = C*16;

    alignas(32) int32_t (&bias)[M];
    alignas(32) int8_t ckk_row[CKK_ROW_SIZE];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    float scale;
	
  public:
    QLinearConvScalarStream (
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    ): bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero(x_zero), w_zero(w_zero), y_zero(y_zero) {
      scale = x_scale*w_scale/y_scale;
    };

		void filter(
			input_window<int8_t>* in,
      input_stream<int8_t>* weights,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QLinearConvScalarStream::filter);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Vector implementation, QLinearConv5x5<28,24,1,1,6,5> takes 3237 cycles.
 * Requires data to be arranged in [a,b,c,d,e] -> [0,0,0,0,a,a,b,b,c,c,d,d,e,e,0,0], 
 * due to int8 indexing restriction. Requires INP_W%16=0, OUT_W_PAD%16=0
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int K>
class QLinearConv5x5 {
  
  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;

    alignas(32) int8_t (&weights)[M*C*K*16];
    alignas(32) int32_t (&bias)[M];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    // precomputation
    int scalebits;
    int16_t scale;
	
  public:
    QLinearConv5x5 (
      int8_t (&w)[M*C*K*16],
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    );

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(K==5);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
			REGISTER_FUNCTION(QLinearConv5x5::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Vector implementation, QLinearConv5x5Scale32bit<28,24,1,1,6,5> takes 7063 cycles.
 * Requires data to be arranged in [a,b,c,d,e] -> [0,0,0,0,a,a,b,b,c,c,d,d,e,e,0,0], 
 * due to int8 indexing restriction. Requires INP_W%16=0, OUT_W_PAD%16=0
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int K>
class QLinearConv5x5Scale32bit {
  
  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;

    alignas(32) int8_t (&weights)[M*C*K*16];
    alignas(32) int32_t (&bias)[M];
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    // precomputation
    int scalebits;
    int32_t scale;
	
  public:
    QLinearConv5x5Scale32bit (
      int8_t (&w)[M*C*K*16],
      int32_t (&b)[M],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    );

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(INP_W%16==0 && OUT_W_PAD%16==0);
			REGISTER_FUNCTION(QLinearConv5x5Scale32bit::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // QLINEARCONV_H_
