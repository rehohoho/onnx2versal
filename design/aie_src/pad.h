#ifndef PAD_H_
#define PAD_H_

#include <type_traits>
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
 * PadVectorInt16::filter<int16_t, 28, 28, 32>, total = 5521
 */
template <typename TT, int N, int INP_W, int OUT_W>
class PadVectorInt16 {
  private:
    static const int WORD_SIZE_BITS = 16;

  public:
    void filter(
      input_window<int16>* in,  // NxINP_W
      output_window<int16>* out // NxOUT_W
    );

    static void registerKernelClass() {
      assert(INP_W<=OUT_W && (std::is_same<TT, int16_t>::value));
      assert((std::is_same<TT, int16>::value));
      REGISTER_FUNCTION(PadVectorInt16::filter);
    }
};
/** @}*/


#endif // PAD_H_
