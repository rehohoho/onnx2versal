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
 * QgemmVector<1,84,16> takes 16966 cycles
 */
template <int M, int K, int N>
class QgemmScalar {
  
  private:
    alignas(32) int8_t (&weights)[N*K]; // KxN (256x120)
    alignas(32) int32_t (&bias)[N];     // N   (120)
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    float scale;
  
  public:
    QgemmScalar (
      int8_t (&w)[K*N],
      int32_t (&b)[N],
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
 * @brief Vector implementation for MK*KN, stores weights and biases, requires N%16=0
 * QgemmVector<1,84,16> takes 125 cycles
 */
template <int M, int K, int N>
class QgemmVector {
  
  private:
    alignas(32) int8_t (&weights)[N*K]; // KxN (256x120)
    alignas(32) int32_t (&bias)[N];     // N   (120)
    float x_scale;
    float w_scale;
    float y_scale;
    int8_t x_zero;
    int8_t w_zero;
    int8_t y_zero;

    // precomputation
    int scalebits;
    int16_t scale;
    int32_t shift;

    static constexpr int K_REM16 = K%16;
    static constexpr int RUN_8CHUNK = K_REM16 >= 8;
    static constexpr int K_REM8 = RUN_8CHUNK ? K_REM16 - 8 : K_REM16;
    static constexpr int RUN_LASTCHUNK = K_REM8 > 0;
  
  public:
    QgemmVector (
      int8_t (&w)[K*N],
      int32_t (&b)[N],
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
      static_assert(N % 16 == 0);
      REGISTER_FUNCTION(QgemmVector::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};
/** @}*/


#endif // QGEMM_H_
