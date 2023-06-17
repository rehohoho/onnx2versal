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
 * AddScalar<float_t, 16384, 1> takes 131102 cycles, (window 45097*8=360776 cycles)
 */
template <typename TT, int W, int IS_RELU>
class AddScalar {
	public:
		void filter(
			input_stream<TT>* restrict inA,
			input_stream<TT>* restrict inB,
			output_stream<TT>* restrict out
		);

		static void registerKernelClass() {
			static_assert(sizeof(TT) == 4);
			REGISTER_FUNCTION(AddScalar::filter);
		}
};


/**
 * @brief Vector implementation for float, 
 * AddScalar<float_t, 16384, 1> takes 28686 cycles (window 5668*8=45344 cycles)
 */
template <typename TT, int W, int IS_RELU>
class AddFloat {
	public:
		void filter(
			input_stream<TT>* restrict inA,
			input_stream<TT>* restrict inB,
			output_stream<TT>* restrict out
		);

		static void registerKernelClass() {
			static_assert(W%4 == 0 && (std::is_same<TT, float>::value));
			REGISTER_FUNCTION(AddFloat::filter);
		}
};
/** @}*/


#endif // ADD_H_
