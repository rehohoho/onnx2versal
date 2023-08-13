#ifndef QLINEARSOFTMAX_H_
#define QLINEARSOFTMAX_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup QLinearSoftmaxKernels
 * @ingroup QLinearSoftmax
 * - qy = saturate ((y / y_scale) + y_zero_point) => x = (qx - qx_zero) * qx_scale
 * - Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * 
 * For exp(x) ~= (1 + x/256)**256,
 * Exp shift is redundant as softmax divides, 
 * Minus max for numerical stability, we have x1 = 1+(x - xmax)/256 = 1 + (qx - xmax)*qx_scale/256
 * Then x' = x1*x1 (8 times)
 * 
 * For y = exp(x)/div, qy = exp(x)/div/qy_scale + qy_zero
 * 
 * @{
 */

/**
 * @brief Scalar implementation.
 * QLinearSoftmaxScalar<10,20,32> takes 517922 cycles for expf, cycles 164026 for fastexp2.
 */
template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
class QLinearSoftmaxScalar {
  
  private:
    float x_scale;
    float y_scale;
    TT x_zero;
    TT y_zero;

    float scale;
  
  public:
    QLinearSoftmaxScalar (
      float x_scale,
      float y_scale,
      TT x_zero,
      TT y_zero
    ): x_scale(x_scale), y_scale(y_scale), x_zero(x_zero), y_zero(y_zero) {};

    void filter(
      input_window<TT>* in,
      output_stream<TT>* out
    );
    
    static void registerKernelClass() {
      static_assert((std::is_same<TT, int8_t>::value) || (std::is_same<TT, uint8_t>::value));
      REGISTER_FUNCTION(QLinearSoftmaxScalar::filter);
    };
};


/**
 * @brief Vector implementation using fastexp2 method, float multiplication for exp estimation
 * QLinearSoftmaxFloatmul<10,10,16> takes 6410 cycles
 * requires INP_W_PAD%16=0.
 */
template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
class QLinearSoftmaxFloatmul {
	
  private:
    float x_scale;
    float y_scale;
    TT x_zero;
    TT y_zero;

    int EXP_BITSHIFT = 18;
    int OUT_BITSHIFT = 10;

    // precompute
    float fastexp_scale;
    int16_t min_value;

  public:
    QLinearSoftmaxFloatmul (
      float x_scale,
      float y_scale,
      TT x_zero,
      TT y_zero
    );

		void filter(
			input_window<TT>* in,
			output_stream<TT>* out
		);

		static void registerKernelClass() {
      static_assert((std::is_same<TT, int8_t>::value) || (std::is_same<TT, uint8_t>::value));
			static_assert(INP_W_PAD % 16 == 0);
			REGISTER_FUNCTION(QLinearSoftmaxFloatmul::filter);
		}
};


/**
 * @brief Vector implementation using fastexp2 method for single axis, 
 * QLinearSoftmaxSingleaxis<10,10,16> takes 5185 cycles
 * requires INP_W_PAD%16=0. Slightly less accurate due to srs after each mult.
 */
template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
class QLinearSoftmaxSingleaxis {
	
  private:
    float x_scale;
    float y_scale;
    TT x_zero;
    TT y_zero;

    int EXP_BITSHIFT;
    int OUT_BITSHIFT = 4;

    // precompute
    int16_t fastexp_scale;
    int32_t expsum_offset;
    int16_t min_value;

  public:
    QLinearSoftmaxSingleaxis (
      float x_scale,
      float y_scale,
      TT x_zero,
      TT y_zero
    );

		void filter(
			input_window<TT>* in,
			output_stream<TT>* out
		);

		static void registerKernelClass() {
      static_assert((std::is_same<TT, int8_t>::value) || (std::is_same<TT, uint8_t>::value));
			static_assert(INP_W_PAD % 16 == 0);
			REGISTER_FUNCTION(QLinearSoftmaxSingleaxis::filter);
		}
};
/** @}*/


#endif // QLINEARSOFTMAX_H_
