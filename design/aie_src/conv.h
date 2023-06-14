#ifndef CONV_H_
#define CONV_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup Conv2DKernels
 * @ingroup Conv2D
 * 
 * @details 
 * Reference performance:
 *  - B*M*C*K*K*OUT_H*OUT_W computations
 * 
 * Design notes for non-padded weights
 * - Using conditionals ~2x loop time, so shuffle down to handle %4 vs %5
 * - Compiler does not detect dependencies across C-loop within W-loop for some C, M, K
 * 
 * Design notes
 *  - Compute constrained
 *  - Loop order BMHW seems faster since H and W > M
 *  - Multiple accs reduces number of loads by reusing data
 *  - Preloading is not useful since it loads extra every C*K*K
 * 
 * @{
 */


/**
 * @brief Scalar implementation for BHWC, stores weights and biases,
 * ConvReluScalarBHWC<28, 28, 24, 1, 1, 6, 5> total = 247757 cycles
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class ConvReluScalarBHWC {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&weights)[M*K*K*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalarBHWC(
      float (&w)[M*K*K*C],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    
    static void registerKernelClass() {
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(ConvReluScalarBHWC::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar implementation for BCHW, stores weights and biases,
 * ConvReluScalarBCHW<28, 28, 24, 1, 1, 6, 5> total = 236147 cycles
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class ConvReluScalarBCHW {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&weights)[M*K*K*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalarBCHW(
      float (&w)[M*K*K*C],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar stream implementation for BCHW, stores biases,
 * ConvReluScalarBCHWStream<28, 28, 24, 1, 1, 6, 5> total = 1359043 cycles
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class ConvReluScalarBCHWStream {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float w_row[OUT_H*OUT_W];

  public:
    ConvReluScalarBCHWStream(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarBCHWStream::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases, requires OUT_W%8=0
 * Conv5x5ReluBCHW<28, 24, 1, 1, 6> total = 21199 cycles
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int _K_notused, int IS_RELU>
class Conv5x5ReluBCHW {

  private:
    static constexpr int K = 5;
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&weights)[M*C*K*K];
    alignas(32) float (&bias)[M];

  public:
    Conv5x5ReluBCHW(
      float (&w)[M*C*K*K],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(OUT_W%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv5x5ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases, requires OUT_W%8=0
 * Conv5x5on8ReluBCHW<28, 24, 1, 1, 6> total = 16521 cycles
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int _K_notused, int IS_RELU>
class Conv5x5on8ReluBCHW {

  private:
    static constexpr int K = 5;
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&weights)[M*C*K*8];
    alignas(32) float (&bias)[M];

  public:
    Conv5x5on8ReluBCHW(
      float (&w)[M*C*K*8],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(OUT_W%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv5x5on8ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};
/** @}*/


#endif // CONV_H_
