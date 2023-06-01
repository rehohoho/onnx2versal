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
 * SoftmaxScalar<10, 10> takes 250322 cycles
 */
template <int INP_H, int INP_W, int INP_W_PAD>
class SoftmaxScalar {
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
