#ifndef QLINEARSOFTMAX_H_
#define QLINEARSOFTMAX_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup QlinearsoftmaxKernels
 * @ingroup Qlinearsoftmax
 * - qy = saturate ((y / y_scale) + y_zero_point) => x = (qx - qx_zero) * qx_scale
 * - Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * 
 * For exp(x) ~= (1 + x/256)**256,
 * Let x1 = 1+x/256 = 1 + (qx-qx_zero)*qx_scale/256 = qx*qx_scale/256 - qx_zero*qx_scale/256 + 1
 * Then x' = x1*x1 (8 times)
 * 
 * For y = exp(x)/div, qy = exp(x)/div/qy_scale + qy_zero
 * 
 * @{
 */

/**
 * @brief Scalar implementation.
 * QlinearsoftmaxScalar<10,20,32> takes 517922 cycles
 */
template <int INP_H, int INP_W, int INP_W_PAD>
class QlinearsoftmaxScalar {
  
  private:
    float x_scale;
    float y_scale;
    int8_t x_zero;
    int8_t y_zero;

    float scale;
  
  public:
    QlinearsoftmaxScalar (
      float x_scale,
      float y_scale,
      int8_t x_zero,
      int8_t y_zero
    ): x_scale(x_scale), y_scale(y_scale), x_zero(x_zero), y_zero(y_zero) {};

    void filter(
      input_window<int8_t>* in,
      output_window<int8_t>* out
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(QlinearsoftmaxScalar::filter);
    };
};


/**
 * @brief Vector implementation, float multiplication for exp estimation
 * QlinearsoftmaxFloatmul<10,10,16> takes 3684 cycles
 * requires INP_W_PAD%16=0.
 */
template <int INP_H, int INP_W, int INP_W_PAD>
class QlinearsoftmaxFloatmul {
	
  private:
    float x_scale;
    float y_scale;
    int8_t x_zero;
    int8_t y_zero;

    int OUT_BITSHIFT = 16;
    int EXP_BITSHIFT = 8;

    // precompute
    float fastexp_scale;
    float fastexp_shift;

  public:
    QlinearsoftmaxFloatmul (
      float x_scale,
      float y_scale,
      int8_t x_zero,
      int8_t y_zero
    );

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			static_assert(INP_W_PAD % 16 == 0);
			REGISTER_FUNCTION(QlinearsoftmaxFloatmul::filter);
		}
};


/**
 * @brief Vector implementation for single axis, 
 * QlinearsoftmaxSingleaxis<10,10,16> takes 2257 cycles
 * requires INP_W_PAD%16=0. Slightly less accurate due to srs after each mult.
 */
template <int INP_H, int INP_W, int INP_W_PAD>
class QlinearsoftmaxSingleaxis {
	
  private:
    float x_scale;
    float y_scale;
    int8_t x_zero;
    int8_t y_zero;

    int EXP_BITSHIFT = 24;
    int OUT_BITSHIFT = 8;

    // precompute
    int16_t fastexp_scale;
    int32_t fastexp_shift;
    int32_t expsum_offset;

  public:
    QlinearsoftmaxSingleaxis (
      float x_scale,
      float y_scale,
      int8_t x_zero,
      int8_t y_zero
    );

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			static_assert(INP_W_PAD % 16 == 0);
			REGISTER_FUNCTION(QlinearsoftmaxSingleaxis::filter);
		}
};
/** @}*/


#endif // QLINEARSOFTMAX_H_
