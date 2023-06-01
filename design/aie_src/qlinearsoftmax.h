#ifndef QLINEARSOFTMAX_H_
#define QLINEARSOFTMAX_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup QlinearsoftmaxKernels
 * @ingroup Qlinearsoftmax
 * - qy = saturate ((y / y_scale) + y_zero_point) => x = (qx - qx_zero) * qx_scale
 * - Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * 
 * @{
 */

/**
 * @brief Scalar implementation.
 * QlinearsoftmaxScalar<10,20,32> takes 109503 cycles
 */
template <int INP_H, int INP_W, int INP_W_PAD>
class QlinearsoftmaxScalar {
  
  private:
    float x_scale;
    float y_scale;
    int8_t x_zero;
    int8_t y_zero;

    float scale;
  
  public:
    QlinearsoftmaxScalar (
      float x_scale,
      float y_scale,
      int8_t x_zero,
      int8_t y_zero
    ): x_scale(x_scale), y_scale(y_scale), x_zero(x_zero), y_zero(y_zero) {};

    void filter(
      input_window<int8_t>* in,
      output_window<int8_t>* out
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(QlinearsoftmaxScalar::filter);
    };
};
/** @}*/


#endif // QLINEARSOFTMAX_H_
