#ifndef QLINEARMAC_KERNEL_H
#define QLINEARMAC_KERNEL_H

#include <type_traits>
#include <assert.h>
#include <adf.h>


/** 
 * @defgroup QlinearMacKernels
 * @ingroup QlinearMac
 * 
 * @details
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
 * - y = saturate ((x / y_scale) + y_zero)
 * - x = (qx - qx_zero) * qx_scale
 * 
 * Computation
 * - qz = (qx-qx_zero)*qx_scale * (qw-qw_zero)*qw_scale / qz_scale + qz_zero
 * - qy = (qz-qz_zero)*qz_scale + (qb-qb_zero)*qb_scale / qy_scale + qy_zero
 * 
 * @{
 */


/**
 * @brief QlinearMac scalar implementation, 
 * QlinearMacScalar<14,128,0> 418101 cycles
 */
template <typename TT, typename TTPARAM, int B, int W, int IS_RELU>
class QlinearMacScalar {
  
  private:
    alignas(32) TTPARAM (&weights)[W];
    alignas(32) TTPARAM (&bias)[W];
    float x_scale;
    float w_scale;
    float b_scale;
    float z_scale;
    float y_scale;
    TT x_zero;
    TTPARAM w_zero;
    TTPARAM b_zero;
    TT z_zero;
    TT y_zero;

    // precompute
    float scale_x;
    float scale_z;
    float shift_x[W];
    float shift_z[W];
	
  public:
    QlinearMacScalar (
      TTPARAM (&w)[W],
      TTPARAM (&b)[W],
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TTPARAM b_zero,
      TT z_zero,
      TT y_zero
    );

		void filter(
			input_window<TT>* in,
      output_window<TT>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(QlinearMacScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};

/**
 * @brief QlinearMac vector implementation,
 * requires W%16 == 0,
 * QlinearMacScalar<14,128,0> 1256 cycles, 2692 for int32 scale
 */
template <typename TT, typename TTPARAM, int B, int W, int IS_RELU>
class QlinearMac {
  
  private:
    alignas(32) TTPARAM (&weights)[W];
    alignas(32) TTPARAM (&bias)[W];
    float x_scale;
    float w_scale;
    float b_scale;
    float z_scale;
    float y_scale;
    TT x_zero;
    TTPARAM w_zero;
    TTPARAM b_zero;
    TT z_zero;
    TT y_zero;

    // precompute
    int bitshift_x;
    int bitshift_z;
    int16_t scale_z;
    alignas(32) int32_t scale_x[W];
    alignas(32) int32_t shift_x[W];
    alignas(32) int32_t shift_z[W];
	
  public:
    QlinearMac (
      TTPARAM (&w)[W],
      TTPARAM (&b)[W],
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TTPARAM b_zero,
      TT z_zero,
      TT y_zero
    );

		void filter(
			input_stream<TT>* in,
			output_stream<TT>* out
		);

		static void registerKernelClass() {
      static_assert(W % 16 == 0);
			REGISTER_FUNCTION(QlinearMac::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
		}
};
/** @}*/


#endif // QLINEARMAC_KERNEL_H
