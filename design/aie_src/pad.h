#ifndef PAD_H_
#define PAD_H_

#include <type_traits>
#include <adf.h>
#include <assert.h>


/** 
 * @defgroup PadKernels
 * @ingroup Pad
 * 
 * @brief See padding at https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
 * 
 * @{
 */

/**
 * @brief Scalar implementation for Pad2D
 * Pad2DScalar<f,2,32,32,1,1,1,1> takes 2964 cycles (tested 954 window, 13k mixed)
 */
template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
class Pad2DScalar {
  private:
    static constexpr int OUT_H = INP_H + H0 + H1;
    static constexpr int OUT_W = INP_W + W0 + W1;
    
  public:
    void filter(
      input_stream<TT>* in,
      output_stream<TT>* out
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(Pad2DScalar::filter);
    }
};


/**
 * @brief Vector implementation for Float Pad2D
 * Pad2DFloat<f,2,32,32,1,1,1,1> takes 2867 cycles
 */
template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
class Pad2DFloat {
  private:
    static constexpr int OUT_H = INP_H + H0 + H1;
    static constexpr int OUT_W = INP_W + W0 + W1;
    
  public:
    void filter(
      input_stream<TT>* in,
      output_stream<TT>* out
    );

    static void registerKernelClass() {
      static_assert((std::is_same<TT, float>::value));
      REGISTER_FUNCTION(Pad2DFloat::filter);
    }
};
/** @}*/


#endif // PAD_H_
