#ifndef QGEMM_H_
#define QGEMM_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup QgemmKernels
 * @ingroup Qgemm
 * 
 * @{
 */

/**
 * @brief Scalar implementation for MK*KN, stores weights and biases,
 * QgemmScalar<1, 84, 10> total = 
 */
template <int M, int K, int N, int NPAD>
class QgemmScalar {
  
  private:
    alignas(32) int8_t (&weights)[NPAD*K]; // KxN (256x120)
    alignas(32) int32_t (&bias)[NPAD];      // N   (120)
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    float scale;
  
  public:
    QgemmScalar (
      int8_t (&w)[K*NPAD],
      int32_t (&b)[NPAD],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    ): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero(x_zero), w_zero(w_zero), y_zero(y_zero) {
      scale = x_scale*w_scale/y_scale;
    };

    void filter(
      input_window<int8_t>* in,   // MxK  (1x256)
      output_window<int8_t>* out  // MxN  (1x120)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(QgemmScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};


/**
 * @brief Vector implementation for MK*KN, stores weights and biases,
 * QgemmVector<1, 84, 10> total = 
 */
template <int M, int K, int N, int NPAD>
class QgemmVector {
  
  private:
    alignas(32) int8_t (&weights)[NPAD*K]; // KxN (256x120)
    alignas(32) int32_t (&bias)[NPAD];      // N   (120)
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    // precomputation
    int16_t scalebits;
    int16_t scale;
    int32_t shift;
  
  public:
    QgemmVector (
      int8_t (&w)[K*NPAD],
      int32_t (&b)[NPAD],
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero
    );

    void filter(
      input_window<int8_t>* in,   // MxK  (1x256)
      output_window<int8_t>* out  // MxN  (1x120)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(QgemmVector::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};
/** @}*/


#endif // QGEMM_H_
