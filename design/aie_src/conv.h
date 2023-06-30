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
 *  - B*M*C*KH*KW*OUT_H*OUT_W_PAD computations
 * 
 * Design notes for non-padded weights
 * - Using conditionals ~2x loop time, so shuffle down to handle %4 vs %5
 * - Compiler does not detect dependencies across C-loop within W-loop for some C
 * 
 * Design notes
 *  - Compute constrained
 *  - Loop order BMHW seems faster since H and W > M
 *  - Multiple accs reduces number of loads by reusing data
 *  - Preloading is not useful since it loads extra every C*KH*KW
 * 
 * @{
 */


/**
 * @brief Scalar implementation for BHWC, stores weights and biases, 
 * requires STEP_H==STEP_W==1, GROUP==1, 
 * ConvReluScalarBHWC<28,28,24,1,1,1,1,4,5,1> total = 147947 cycles
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvReluScalarBHWC {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    alignas(32) float (&weights)[M*KH*KW*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalarBHWC(
      float (&w)[M*KH*KW*C],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    
    static void registerKernelClass() {
      static_assert(GROUP == 1);
      static_assert(STEP_H == 1);
      static_assert(STEP_W == 1);
      REGISTER_FUNCTION(ConvReluScalarBHWC::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar implementation for BCHW, stores weights and biases,
 * requires GROUP==1, 
 * ConvReluScalarBCHW<28,28,24,1,1,1,1,4,5,5,1,1> total = 140059
 * ConvReluScalarBCHW<26,26,24,1,1,1,1,4,3,3,1,1> total = 65443
 * ConvReluScalarBCHW<24,24,10,2,2,1,1,4,5,5,1,1> total = 24798
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvReluScalarBCHW {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    alignas(32) float (&weights)[M*KH*KW*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalarBCHW(
      float (&w)[M*KH*KW*C],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(C % GROUP == 0);
      REGISTER_FUNCTION(ConvReluScalarBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases, 
 * requires KH==KW==5, INP_W%4==0, OUT_W_PAD%8=0, STEP_H==STEP_W==1, GROUP==1, 
 * Conv5x5ReluBCHW<28,28,24,1,1,1,1,4,5,1> total = 14148
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv5x5ReluBCHW {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    alignas(32) float (&weights)[M*C*KH*KW];
    alignas(32) float (&bias)[M];

  public:
    Conv5x5ReluBCHW(
      float (&w)[M*C*KH*KW],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(GROUP == 1);
      static_assert(KH==5);
      static_assert(KW==5);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv5x5ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 5x5 BCHW, stores weights and biases, 
 * requires KH==KW==5, INP_W%4==0, OUT_W_PAD%8=0, STEP_H==STEP_W==1, GROUP==1, 
 * assumes weights are padded to MxCx5x8,
 * Conv5x5on8ReluBCHW<28,28,24,1,1,1,1,4,5,1> total = 11030
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv5x5on8ReluBCHW {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    alignas(32) float (&weights)[M*C*KH*8];
    alignas(32) float (&bias)[M];

  public:
    Conv5x5on8ReluBCHW(
      float (&w)[M*C*KH*8],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(GROUP == 1);
      static_assert(KH==5);
      static_assert(KW==5);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv5x5on8ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector implementation for 3x3 BCHW, stores weights and biases, 
 * requires KH==KW==3, INP_W%4==0, OUT_W_PAD%8=0, STEP_H==1, STEP_W==1, GROUP==1, 
 * assumes weights are padded to MxCx12,
 * Conv3x3on12ReluBCHW<26,28,24,1,1,1,1,4,3,1> start = 421518,end = 426929,total = 5411 (5363)
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv3x3on12ReluBCHW {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
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
      static_assert(GROUP == 1);
      static_assert(KH==3);
      static_assert(KW==3);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(Conv3x3on12ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar stream implementation for BCHW, stores biases,
 * requires GROUP==1, 
 * ConvReluScalarStreamCacheCKK<26,28,24,1,1,1,1,4,3,1> total = 74845
 * ConvReluScalarStreamCacheCKK<24,24,11,2,2,1,1,4,3,1> total = 16863
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvReluScalarStreamCacheCKK {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[C/GROUP*KH*KW];

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
      static_assert(C % GROUP == 0);
      REGISTER_FUNCTION(ConvReluScalarStreamCacheCKK::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires KH==KW==3, INP_W%4==0, OUT_W_PAD%(8|4)==0, STEP_H==1|2, STEP_W==1|2, GROUP==1, 
 * Conv3x3ReluStreamCacheCKK<26,28,24,1,1,1,1,4,3,1> total = 11059
 * Conv3x3ReluStreamCacheCKK<24,24,12,2,2,1,1,4,3,1> total = 3656
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv3x3ReluStreamCacheCKK {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
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
      static_assert(GROUP == 1);
      static_assert(KH==3);
      static_assert(KW==3);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0 && STEP_W==1 || OUT_W_PAD%4==0 && STEP_W==2);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
      REGISTER_FUNCTION(Conv3x3ReluStreamCacheCKK::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires KH==KW==3, INP_W%4==0, OUT_W_PAD%8==0, STEP_H==1, STEP_W==1, GROUP==1, 
 * Conv3x3ReluStreamCacheCKKMultiRow<26,28,24,1,1,1,1,4,3,1> total = 10672
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv3x3ReluStreamCacheCKKMultiRow {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[C*12];
    alignas(32) float out_row[OUT_W_PAD];

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
      static_assert(GROUP == 1);
      static_assert(KH==3);
      static_assert(KW==3);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0);
      static_assert(STEP_H == 1);
      static_assert(STEP_W == 1);
      REGISTER_FUNCTION(Conv3x3ReluStreamCacheCKKMultiRow::filter);
      REGISTER_PARAMETER(bias);
    }

};
/** @}*/


#endif // CONV_H_
