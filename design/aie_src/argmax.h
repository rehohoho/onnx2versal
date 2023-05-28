#ifndef ARGMAX_H_
#define ARGMAX_H_

#include <adf.h>


/** 
 * @defgroup ArgmaxKernels
 * @ingroup Argmax
 * 
 * @{
 */


/**
 * @brief Scalar implementation, ArgmaxScalar<100, 10> takes 1064 cycles
 */
template <int CHUNK_CNT, int CHUNK_SIZE>
class ArgmaxScalar {
	public:
		void filter(
			input_window<float>* in,
			output_window<float>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(ArgmaxScalar::filter);
		}
};
/** @}*/


#endif // ARGMAX_H_
