#ifndef QLINEARADD_H_
#define QLINEARADD_H_

#include <assert.h>
#include <type_traits>
#include <adf.h>


/** 
 * @defgroup QLinearAddKernels
 * @ingroup QLinearAdd
 * 
 * @brief 
 * See https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add, 
 * See https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear
 * 
 * @{
 */


/**
 * @brief Scalar implementation, 
 * requires W%16==0,
 * QLinearAddInt8<float_t, 16384, 1> takes 8231 cycles
 */
template <typename TT, int W, int IS_RELU>
class QLinearAddInt8 {
	private:
    float a_scale;
    float b_scale;
    float c_scale;
    int8_t a_zero;
    int8_t b_zero;
    int8_t c_zero;

		int bitshift;
		int32_t ascale;
		int32_t bscale;
		int32_t shiftv;
	
	public:
		QLinearAddInt8 (
      float a_scale,
			float b_scale,
			float c_scale,
			int8_t a_zero,
			int8_t b_zero,
			int8_t c_zero
    );

		void filter(
			input_stream<TT>* restrict inA,
			input_stream<TT>* restrict inB,
			output_stream<TT>* restrict out
		);

		static void registerKernelClass() {
			static_assert(W%16==0);
			static_assert((std::is_same<TT, int8_t>::value));
			REGISTER_FUNCTION(QLinearAddInt8::filter);
		}
};
/** @}*/


#endif // QLINEARADD_H_
