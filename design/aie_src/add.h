#ifndef ADD_H_
#define ADD_H_

#include <adf.h>


/** 
 * @defgroup AddKernels
 * @ingroup Add
 * 
 * @{
 */


/**
 * @brief Scalar implementation, AddScalar<1, 16384> takes cycles
 */
template <typename TT, int B, int W, int IS_RELU>
class AddScalar {
	public:
		void filter(
			input_window<TT>* inA,
			input_window<TT>* inB,
			output_window<TT>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(AddScalar::filter);
		}
};
/** @}*/


#endif // ADD_H_
