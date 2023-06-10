#ifndef QLINEARMAC_KERNEL_H
#define QLINEARMAC_KERNEL_H

#include <type_traits>
#include <assert.h>
#include <adf.h>


/** 
 * @defgroup QlinearMacKernels
 * @ingroup QlinearMac
 * - y = saturate ((x / y_scale) + y_zero)
 * - x = (qx - qx_zero) * qx_scale
 * 
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
 * y = max(x * tw + tbias, 0)
 * 
 * @{
 */


/**
 * @brief QlinearMac scalar implementation
 */
template <int B, int W, int IS_RELU>
class QlinearMacScalar {
  
  private:
    alignas(32) int8_t (&weights)[W];
    alignas(32) int8_t (&bias)[W];
    float x_scale;
    float w_scale;
    float b_scale;
    float z_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t b_zero;
    int8_t z_zero;
    int8_t y_zero;
	
  public:
    QlinearMacScalar (
      int8_t (&w)[W],
      int8_t (&b)[W],
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t b_zero,
      int8_t z_zero,
      int8_t y_zero
    ): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), b_scale(b_scale), z_scale(z_scale), y_scale(y_scale), x_zero(x_zero), w_zero(w_zero), b_zero(b_zero), z_zero(z_zero), y_zero(y_zero){};

		void filter(
			input_window<int8_t>* in,
			output_window<int8_t>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QlinearMacScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // QLINEARMAC_KERNEL_H
