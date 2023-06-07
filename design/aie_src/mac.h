#ifndef MAC_KERNEL_H
#define MAC_KERNEL_H

#include <type_traits>
#include <assert.h>
#include <adf.h>


/** 
 * @defgroup MacKernels
 * @ingroup Mac
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
 * y = max(x * tw + tbias, 0)
 * 
 * @{
 */


/**
 * @brief Scalar implementation,
 * MacScalar<196,128> takes cycles
 */
template <typename TT, int B, int W, int IS_RELU>
class MacScalar {
  
  private:
    alignas(32) TT (&weights)[W];
    alignas(32) TT (&bias)[W];
	
  public:
    MacScalar (
      TT (&w)[W],
      TT (&b)[W]
    ): weights(w), bias(b) {};

		void filter(
			input_window<TT>* in,
			output_window<TT>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(MacScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};


/**
 * @brief Scalar implementation,
 * MacFloat<196,128> takes 1881 cycles
 */
template <typename TT, int B, int W, int IS_RELU>
class MacFloat {
  
  private:
    alignas(32) float (&weights)[W];
    alignas(32) float (&bias)[W];
	
  public:
    MacFloat (
      float (&w)[W],
      float (&b)[W]
    ): weights(w), bias(b) {};

		void filter(
			input_window<float>* in,
			output_window<float>* out
		);

		static void registerKernelClass() {
      static_assert(W % 8 == 0 && (std::is_same<TT, float>::value));
			REGISTER_FUNCTION(MacFloat::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // MAC_KERNEL_H
