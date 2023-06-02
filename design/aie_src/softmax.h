#ifndef SOFTMAX_H_
#define SOFTMAX_H_

#include <adf.h>


/** 
 * @defgroup SoftmaxKernels
 * @ingroup Softmax
 * - Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * 
 * @{
 */


/**
 * @brief Scalar implementation, 
 * SoftmaxScalar<10, 10> takes 
 * - expf: 250322 cycles, within max abs diff 1e-05, max rel diff 1e-03
 * - fastexp: 22200 cycles, max abs diff: 0.0046130717, max rel diff: 0.039221319418778974
 * - fastexp2: 68282 cycles, max abs diff: 0.0002221614, max rel diff: 0.00129
 * - fastexp3: 10484 cycles, max abs diff: 0.0010825096, max rel diff: 0.00956
 */
template <int INP_H, int INP_W, int INP_W_PAD>
class SoftmaxScalar {
	private:
	  float coef[10] = {1, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333, 0.001388888888888889, 0.0001984126984126984, 2.48015873015873e-05, 2.7557319223985893e-06, 2.755731922398589e-07};
		float fastexp3(float val, int precision);

	public:
		void filter(
			input_window<float>* in,
			output_window<float>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(SoftmaxScalar::filter);
		}
};
/** @}*/


#endif // SOFTMAX_H_
