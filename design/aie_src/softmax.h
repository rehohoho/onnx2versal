#ifndef SOFTMAX_H_
#define SOFTMAX_H_

#include <adf.h>


/** 
 * @defgroup SoftmaxKernels
 * @ingroup Softmax
 * 
 * @{
 */


/**
 * @brief Scalar implementation, SoftmaxScalar<10, 10> takes  cycles
 */
template <int CHUNK_COUNT, int CHUNK_SIZE>
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
