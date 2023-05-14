#ifndef CONV_H_
#define CONV_H_

#include <adf.h>


/** 
 * @defgroup Conv2DKernels
 * @ingroup Conv2D
 * 
 * @details 
 * Reference performance
 *  - Lenet Conv2 Tutorial (1x16x12x12 int8) example with matmul ~2k cycles
 *  - Theoretical limit: (16*6*5*5*8*8 + 16*8*8) / 4 = 38656 cycles
 * 
 * For non-padded
 * - Note zstart must be a compile time constant
 * - Using conditionals ~2x loop time, so shuffle down to handle %4 vs %5
 * - Compiler does not detect dependencies across C-loop within W-loop for some C, M, K
 * - With chess_separator_scheduler, 40796 -> 41580
 * 
 * Design decisions
 *  - Compute constrained, no point using extra v8float acc to utilize data[12:16]
 *  - Loop order BMHW seems faster since H and W > M
 *  - Multiple accs reduces number of loads by reusing data
 *  - Unrolling is not useful, must preload extra every C*K*K, not worth (46244 -> 53541)
 *  - Reduce data instances to reduce spill (v32float -> v16float: 43981 -> 40796)
 * 
 * @attention Vector versions assume OUT_W%8=0
 * @{
 */

/**
 * @brief Scalar implementation for BHWC, streams weights and biases, 
 * ConvReluScalarGmemParamBHWC<28, 24, 1, 2, 2, 5> total = 1382714 cycles
 */
template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarGmemParamBHWC {
  public:
    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      input_window<float>* weight,  // MKKC (6x5x5x1)
      input_window<float>* bias,    // M    (6)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarGmemParamBHWC::filter);
    }
};


/**
 * @brief Scalar implementation for BHWC, stores weights and biases,
 * ConvReluScalarBHWC<28, 24, 1, 2, 2, 5> total = 144244 cycles
 */
template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarBHWC {

  private:
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
      REGISTER_FUNCTION(ConvReluScalarBHWC::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar implementation for BCHW, stores weights and biases,
 * ConvReluScalarBCHW<28, 24, 1, 2, 2, 5> total = 148375 cycles
 */
template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarBCHW {

  private:
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
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases,
 * Conv5x5ReluBCHW<28, 24, 1, 2, 2> total = 16241 cycles
 */
template <int INP_W, int OUT_W, int B, int C, int M, int _K_notused>
class Conv5x5ReluBCHW {

  private:
    static const int K = 5;
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
      REGISTER_FUNCTION(Conv5x5ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases,
 * Conv5x5on8ReluBCHW<28, 24, 1, 2, 2> total = 10687 cycles
 */
template <int INP_W, int OUT_W, int B, int C, int M, int _K_notused>
class Conv5x5on8ReluBCHW {

  private:
    static const int K = 5;
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
      REGISTER_FUNCTION(Conv5x5on8ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};
/** @}*/


#endif // CONV_H_
