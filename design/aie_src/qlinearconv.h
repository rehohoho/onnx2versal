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
 * - each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1,C_PER_M,H,W) for kernels that allow GROUP != 1
 */


/**
 * @brief Scalar implementation, 
 * QLinearConvScalar<30,32,28,32,1,1,1,1,6,5> total = 1282213, 
 * QLinearConvScalar<28,32,28,32,1,1,1,1,6,3> total = 867213, 
 * QLinearConvScalar<26,32,28,32,1,1,1,1,6,1> total = 625226, 
 * QLinearConvScalar<26,28,13,16,2,2,1,1,6,3> total = 189225
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConvScalar {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP;

    alignas(32) int8_t (&weights)[M*C_PER_M*KH*KW];
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
      int8_t (&w)[M*C*KH*KW],
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
 * @brief Vector implementation for QLinearConv 5x5, 
 * requires data to be arranged in [a,b,c,d,e] -> [0,0,0,0,a,a,b,b,c,c,d,d,e,e,0,0], 
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * requires INP_W%16=0, OUT_W_PAD%16=0, 
 * QLinearConv5x5<30,32,28,32,1,1,1,1,6,5> total = 3513
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConv5x5 {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;

    alignas(32) int8_t (&weights)[M*C*KH*16];
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
      int8_t (&w)[M*C*KH*16],
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
      static_assert(KH==5);
      static_assert(KW==5);
      static_assert(GROUP == 1);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
			REGISTER_FUNCTION(QLinearConv5x5::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Vector implementation for 3x3 QLinearConv, 
 * requires data to be arranged in [a,b,c,d,e] -> [0,0,0,0,a,a,b,b,c,c,d,d,e,e,0,0], 
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * requires INP_W%16=0, OUT_W_PAD%16=0, 
 * QLinearConv5x5Scale32bit<30,32,28,32,1,1,1,1,6,5> total = 7652
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConv5x5Scale32bit {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;

    alignas(32) int8_t (&weights)[M*C*KH*16];
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
      int8_t (&w)[M*C*KH*16],
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
      static_assert(GROUP == 1);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
			REGISTER_FUNCTION(QLinearConv5x5Scale32bit::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};

/**
 * @brief Vector implementation for 3x3 QLinearConv, 
 * requires data to be arranged in [a,b,c,d,e,f,g,h,i] -> [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0], 
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * requires INP_W%16=0, OUT_W_PAD%16=0, 
 * QLinearConv3x3<28,32,28,32,1,1,1,1,6,3> total = 2906
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConv3x3 {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;

    alignas(32) int8_t (&weights)[M*C*16];
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
    QLinearConv3x3 (
      int8_t (&w)[M*C*16],
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
      static_assert(KH==3);
      static_assert(KW==3);
      static_assert(GROUP == 1);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
			REGISTER_FUNCTION(QLinearConv3x3::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Scalar implementation streaming weights, 
 * requires weights stream to be padded from MxCxKxK to MxCx16, KH=KW < 5,
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * QLinearConvScalarStream<28,32,28,32,1,1,1,1,6,3> total = 776955 (output_window 682564)
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConvScalarStream {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP;
    static constexpr int CKK_ROW_SIZE = C_PER_M*(KH*KW+15)/16*16;

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
			output_stream<int8_t>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QLinearConvScalarStream::filter);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Vector implementation for 3x3 QLinearConv, 
 * requires data to be arranged in [a,b,c,d,e,f,g,h,i] -> [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0], 
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * requires KH==KW==3, INP_W%16=0, OUT_W_PAD%16=0, STEP_H==1|2, STEP_W==1|2, 
 * QLinearConvHx4Stream<28,48,28,32,1,1,1,1,8,3> total = 4689 (output_window slightly faster ~0.85x time), 
 * QLinearConvHx4Stream<26,32,13,16,2,2,1,1,8,3> total = 3376
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConvHx4Stream {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP;
    static constexpr int CKK_ROW_SIZE = C_PER_M*(KH*KW+15)/16*16;

    alignas(32) int32_t (&bias)[M];
    alignas(32) int8_t ckk_row[CKK_ROW_SIZE];
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
    QLinearConvHx4Stream (
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
      input_stream<int8_t>* weights,
			output_stream<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(KW<=4);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
			REGISTER_FUNCTION(QLinearConvHx4Stream::filter);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Vector implementation for 3x3 QLinearConv, padding with y_zero, 
 * requires data to be arranged in (M,C,KH,KW) -> (M,C,KH',4) where KH' = KH*4 padded to nearest 16, e.g. [a,b,c,d,e,f,g,h,i] -> [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0], 
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * requires KW<=3, INP_W%16=0, OUT_W_PAD%16=0, STEP_H==1|2, STEP_W==1|2, 
 * QLinearConvHx4StreamPad<28,48,28,32,1,1,1,1,8,3> total = 5373
 * QLinearConv3x3StreamPad<58,32,5,16,2,2,1,1,64,10,4,1> total = 105984
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConvHx4StreamPad {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP;
    static constexpr int CKK_ROW_SIZE = C_PER_M*(KH*KW+15)/16*16;

    alignas(32) int32_t (&bias)[M];
    alignas(32) int8_t ckk_row[CKK_ROW_SIZE];
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
    QLinearConvHx4StreamPad (
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
      input_stream<int8_t>* weights,
			output_stream<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(KW<=4);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
			REGISTER_FUNCTION(QLinearConvHx4StreamPad::filter);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Vector implementation for 3x3 QLinearConv using 32bit scale for precision, 
 * requires data to be arranged in [a,b,c,d,e,f,g,h,i] -> [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0], 
 * requires bias to be shifted, i.e. tbias - tw.reshape(M,-1).sum(1) * X_zero_point, 
 * requires KH==KW==3, INP_W%16=0, OUT_W_PAD%16=0, STEP_H==1|2, STEP_W==1|2, 
 * QLinearConvHx4StreamScale32bit<28,48,28,32,1,1,1,1,8,3> total = 7408, 
 * QLinearConvHx4StreamScale32bit<26,32,13,16,2,2,1,1,8,3> total = 9329
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConvHx4StreamScale32bit {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP;
    static constexpr int CKK_ROW_SIZE = C_PER_M*(KH*KW+15)/16*16;

    alignas(32) int32_t (&bias)[M];
    alignas(32) int8_t ckk_row[CKK_ROW_SIZE];
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
    QLinearConvHx4StreamScale32bit (
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
      input_stream<int8_t>* weights,
			output_stream<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(KW<=4);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
			REGISTER_FUNCTION(QLinearConvHx4StreamScale32bit::filter);
      REGISTER_PARAMETER(bias);
		}
};

/**
 * @brief Vector implementation for 3x3 QLinearConv, 
 * requires data to be reshaped from (M,C,1,1) to (M,C') where C' is padded to next multiple of 16, 
 * requires bias to be shifted, i.e. tbias - tw_1x1.reshape(M,-1).sum(1) * X_zero_point, 
 * requires KH==KW==1, INP_W%16=0, OUT_W_PAD%16=0, STEP_H==1|2, STEP_W==1|2, 
 * QLinearConv1x1Stream<26,32,28,32,1,1,1,1,6,1> total = 2188
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
class QLinearConv1x1Stream {
  
  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int CKK_ROW_SIZE = (C+15)/16*16;

    alignas(32) int32_t (&bias)[M];
    alignas(32) int8_t ckk_row[CKK_ROW_SIZE];
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
    QLinearConv1x1Stream (
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
      input_stream<int8_t>* weights,
			output_stream<int8_t>* out
		);

		static void registerKernelClass() {
      static_assert(KH==1);
      static_assert(KW==1);
      static_assert(GROUP == 1);
      static_assert(INP_W%16==0);
      static_assert(OUT_W_PAD%16==0);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
			REGISTER_FUNCTION(QLinearConv1x1Stream::filter);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // QLINEARCONV_H_
