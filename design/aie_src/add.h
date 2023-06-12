#ifndef ADD_H_
#define ADD_H_

#include <assert.h>
#include <type_traits>
#include <adf.h>


/** 
 * @defgroup AddKernels
 * @ingroup Add
 * 
 * @brief See https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * 
 * @{
 */


/**
 * @brief Scalar implementation, 
 * AddScalar<float_t, 2048, 1> takes 45097 cycles
 */
template <typename TT, int W, int IS_RELU>
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


/**
 * @brief Vector implementation for float, 
 * AddScalar<float_t, 2048, 1> takes 5668 cycles
 */
template <typename TT, int W, int IS_RELU>
class AddFloat {
	public:
		void filter(
			input_window<TT>* inA,
			input_window<TT>* inB,
			output_window<TT>* out
		);

		static void registerKernelClass() {
			static_assert(W%4 == 0 && (std::is_same<TT, float>::value));
			REGISTER_FUNCTION(AddFloat::filter);
		}
};
/** @}*/


#endif // ADD_H_
