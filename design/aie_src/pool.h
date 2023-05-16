#ifndef POOL_H_
#define POOL_H_

#include <adf.h>


/** 
 * @defgroup Pool2DKernels
 * @ingroup Pool2D
 * 
 * @details
 * Design Notes
 * - Only up to 64 floats in vector registers
 * - Definitely bandwidth limited
 * - 2 accs will blow the 64-float vector regs (901 -> 1330 cycles)
 * - Using window_incr instead of pointers (901 -> 988 cycles)
 * - Concat adds additional computation (901 -> 1468 cycles)
 * - All kernels assume INP_W divisible by OUT_W
 * 
 * @{
 */

/**
 * @brief Scalar implementation for BHWC.
 * MaxpoolScalarBHWC::filter<24, 12, 1, 6> total = 7673
 */
template <int INP_W, int OUT_W, int B, int C>
class MaxpoolScalarBHWC {
  public:
    void filter(
      input_window<float>* in,      // BHWC (1x24x24x6)
      output_window<float>* out     // BPQC (1x12x12x6)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBHWC::filter);
    }
};


/**
 * @brief Scalar implementation for BHWC.
 * MaxpoolScalarBCHW::filter<24, 12, 1, 6> total = 11302
 */
template <int INP_W, int OUT_W, int B, int C>
class MaxpoolScalarBCHW {
  public:
    void filter(
      input_window<float>* in,      // BCHW (1x6x24x24)
      output_window<float>* out     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBCHW::filter);
    }
};


/**
 * @brief Vector implementation for BCHW with 2x2 kernel.
 * Requires OUT_W%4=0.
 * Maxpool2x2BCHW::filter<24, 12, 1, 6> total = 901
 */
template <int INP_W, int OUT_W, int B, int C>
class Maxpool2x2BCHW {
  public:
    void filter(
      input_window<float>* in_window,      // BCHW (1x6x24x24)
      output_window<float>* out_window     // BCPQ (1x6x12x12)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(Maxpool2x2BCHW::filter);
    }
};
/** @}*/


#endif // POOL_H_
