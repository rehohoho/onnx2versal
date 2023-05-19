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
 * PadScalar::filter<INP_W, OUT_W>, total = 373 takes 
 */
template <typename TT, int N, int INP_W, int OUT_W>
class PadScalar {
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
/** @}*/


#endif // PAD_H_
