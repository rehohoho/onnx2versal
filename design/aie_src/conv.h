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
 * ConvReluScalarBHWC<28,28,24,1,1,1,1,4,5,1> total = 180055 cycles
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
 * ConvReluScalarBCHW<28,28,24,1,1,1,1,4,5,1> total = 157445
 * ConvReluScalarBCHW<26,26,24,1,1,1,1,4,3,1> total = 66069
 * ConvReluScalarBCHW<24,24,10,2,2,1,1,4,5,1> total = 27577
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
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases, 
 * requires INP_W%4==0 and OUT_W%8=0 and STEP_H==1 and STEP_W==1
 * Conv5x5ReluBCHW<28,28,24,1,1,1,1,4,5,1> total = 14149
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class Conv5x5ReluBCHW {

  private:
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
      static_assert(K==5);
      static_assert(INP_W%4==0);
      static_assert(OUT_W%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv5x5ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases, 
 * requires INP_W%4==0 and OUT_W%8=0 and STEP_H==1 and STEP_W==1,
 * assumes weights are padded to MxCx5x8,
 * Conv5x5on8ReluBCHW<28,28,24,1,1,1,1,4,5,1> total = 11030
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class Conv5x5on8ReluBCHW {

  private:
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
      static_assert(K==5);
      static_assert(INP_W%4==0);
      static_assert(OUT_W%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv5x5on8ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 3x3 BCHW, stores weights and biases, 
 * requires K==3 and INP_W%4==0 and OUT_W%8=0 and STEP_H==1 and STEP_W==1
 * assumes weights are padded to MxCx12,
 * Conv3x3on12ReluBCHW<26,28,24,1,1,1,1,4,3,1> total = 4860
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class Conv3x3on12ReluBCHW {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&weights)[M*C*12];
    alignas(32) float (&bias)[M];

  public:
    Conv3x3on12ReluBCHW(
      float (&w)[M*C*12],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(K==3);
      static_assert(INP_W%4==0);
      static_assert(OUT_W%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv3x3on12ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar stream implementation for BCHW, stores biases,
 * ConvReluScalarStreamCacheHW<26,26,24,1,1,1,1,4,3,1> total = 342293
 * ConvReluScalarStreamCacheHW<24,24,11,2,2,1,1,4,3,1> total = 71463
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class ConvReluScalarStreamCacheHW {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float w_row[OUT_H*OUT_W];

  public:
    ConvReluScalarStreamCacheHW(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarStreamCacheHW::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar stream implementation for BCHW, stores biases,
 * ConvReluScalarStreamCacheCKK<26,28,24,1,1,1,1,4,3,1> total = 74270
 * ConvReluScalarStreamCacheCKK<24,24,11,2,2,1,1,4,3,1> total = 16600
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class ConvReluScalarStreamCacheCKK {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[C*K*K];

  public:
    ConvReluScalarStreamCacheCKK(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalarStreamCacheCKK::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires K==3, INP_W%4==0, OUT_W%(8 or 4)==0, STEP_H==1 or 2, STEP_W==1 or 2
 * Conv3x3ReluStreamCacheCKK<26,28,24,1,1,1,1,4,3,1> total = 10097
 * Conv3x3ReluStreamCacheCKK<24,24,12,2,2,1,1,4,3,1> total = 3656
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class Conv3x3ReluStreamCacheCKK {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[C*12];

  public:
    Conv3x3ReluStreamCacheCKK(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(K == 3);
      static_assert(INP_W%4==0);
      static_assert(OUT_W%8==0 && STEP_W==1 || OUT_W%4==0 && STEP_W==2);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
      REGISTER_FUNCTION(Conv3x3ReluStreamCacheCKK::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires K==3, INP_W%4==0, OUT_W%8==0, STEP_H==1, STEP_W==1
 * Conv3x3ReluStreamCacheCKKMultiRow<26,28,24,1,1,1,1,4,3,1> total = 9823
 */
template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
class Conv3x3ReluStreamCacheCKKMultiRow {

  private:
    static constexpr int OUT_H = (INP_H - K) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[C*12];
    alignas(32) float out_row[OUT_W*2];

  public:
    Conv3x3ReluStreamCacheCKKMultiRow(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(K == 3);
      static_assert(INP_W%4==0);
      static_assert(OUT_W%8==0);
      static_assert(STEP_H == 1);
      static_assert(STEP_W == 1);
      REGISTER_FUNCTION(Conv3x3ReluStreamCacheCKKMultiRow::filter);
      REGISTER_PARAMETER(bias);
    }

};
/** @}*/


#endif // CONV_H_
