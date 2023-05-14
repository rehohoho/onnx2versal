#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include <adf.h>


/** 
 * @defgroup TransposeKernels
 * @ingroup Transpose
 * 
 * @{
 */

/**
 * @brief Scalar implementation for BHWC to BCHW,
 * TransposeScalarBHWC2BCHW::filter<1, 4, 4, 16>, total = 373
 */
template <int B, int H, int W, int C>
class TransposeScalarBHWC2BCHW {
  public:
    void filter(
      input_window<float>* in,      // BHWC (1x4x4x16)
      output_window<float>* out     // BCHW (1x16x4x4)
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(TransposeScalarBHWC2BCHW::filter);
    }
};
/** @}*/


#endif // TRANSPOSE_H_
