#ifndef PAD_H_
#define PAD_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup PadKernels
 * @ingroup Pad
 * 
 * @brief Pads NxINP_W to NxOUT_W shape. Requires INP_W<=OUT_W.
 * 
 * @{
 */

/**
 * @brief Scalar implementation for Pad
 * PadScalar::filter<int16_t, 28, 28, 32>, total = 5521
 */
template <typename TT, int N, int INP_W, int OUT_W>
class PadScalar {
  private:
    static const int WORD_SIZE_BYTES = 16;
    
  public:
    void filter(
      input_window<TT>* in,  // NxINP_W
      output_window<TT>* out // NxOUT_W
    );

    static void registerKernelClass() {
      assert(INP_W<=OUT_W);
      REGISTER_FUNCTION(PadScalar::filter);
    }
};


/**
 * @brief Vector implementation for Pad
 * PadVector::filter<int16_t, 28, 28, 32>, total = 5521
 */
template <typename TT, int N, int INP_W, int OUT_W>
class PadVector {
  private:
    static const int WORD_SIZE_BITS = 16;

  public:
    void filter(
      input_window<TT>* in,  // NxINP_W
      output_window<TT>* out // NxOUT_W
    );

    static void registerKernelClass() {
      assert(INP_W<=OUT_W);
      REGISTER_FUNCTION(PadVector::filter);
    }
};
/** @}*/


#endif // PAD_H_
