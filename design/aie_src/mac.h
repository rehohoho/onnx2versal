#ifndef MAC_KERNEL_H
#define MAC_KERNEL_H

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
template <typename TT, int B, int INP_W, int IS_RELU>
class MacScalar {
  
  private:
    alignas(32) TT (&weights)[INP_W];
    alignas(32) TT (&bias)[INP_W];
	
  public:
    MacScalar (
      TT (&w)[INP_W],
      TT (&b)[INP_W]
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
/** @}*/


#endif // MAC_KERNEL_H
