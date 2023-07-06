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
 * @brief Scalar implementation for BCHW, stores weights and biases,
 * requires GROUP==1, 
 * ConvReluScalarBCHW<28,28,24,24,1,1,1,2,4,5,5,1,1> total = 369505
 * ConvReluScalarBCHW<24,24,10,10,2,2,1,2,4,5,5,1,1> total = 64413
 * ConvReluScalarBCHW<26,26,24,24,1,1,1,2,4,3,3,1,1> total = 148320
 *
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvReluScalarBCHW {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP; // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1,C_PER_M,H,W)
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
 * Conv5x5ReluBCHW<28,28,24,24,1,1,1,2,4,5,5,1,1> total = 5519
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
 * Conv5x5on8ReluBCHW<28,28,24,24,1,1,1,2,4,5,5,1,1> total = 21401
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
 * requires KW<=4, INP_W%4==0, OUT_W_PAD%8=0, STEP_H==1, STEP_W==1, GROUP==1, 
 * assumes weights are padded to MxCx12,
 * ConvHx4ReluBCHW<26,28,24,24,1,1,1,2,4,3,3,1,1> total = 8899
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvHx4ReluBCHW {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int CKK_ROW_SIZE = C*(KH*KW+3)/4*4;

    alignas(32) float (&weights)[M*CKK_ROW_SIZE];
    alignas(32) float (&bias)[M];

  public:
    ConvHx4ReluBCHW(
      float (&w)[M*CKK_ROW_SIZE],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(GROUP == 1);
      static_assert(KW<=4);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0);
      static_assert(STEP_H == 1 && STEP_W == 1);
      REGISTER_FUNCTION(ConvHx4ReluBCHW::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores weights and biases,
 * requires KH==KW==1, INP_W%4==0, OUT_W_PAD%(8|4)==0, STEP_H==1|2, STEP_W==1|2, GROUP==1, 
 * Conv1x1Relu<24,24,24,24,1,1,1,2,4,1,1,1,1> total = 
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv1x1Relu {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;

    alignas(32) float (&weights)[M*C];
    alignas(32) float (&bias)[M];

  public:
    Conv1x1Relu(
      float (&w)[M*C],
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      output_window<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(GROUP == 1);
      static_assert(KH==1);
      static_assert(KW==1);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0 && STEP_W==1 || OUT_W_PAD%4==0 && STEP_W==2);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
      REGISTER_FUNCTION(Conv1x1Relu::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Scalar stream implementation for BCHW, stores biases,
 * requires GROUP==1, 
 * ConvReluScalarStream<26,28,24,24,1,1,1,2,4,3,3,1,1> total = 252185
 * ConvReluScalarStream<24,24,11,12,2,2,1,2,4,3,3,1,1> total = 54420
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvReluScalarStream {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP; // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1,C_PER_M,H,W)
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[C/GROUP*KH*KW];

  public:
    ConvReluScalarStream(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(C % GROUP == 0);
      REGISTER_FUNCTION(ConvReluScalarStream::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires KW<=3, INP_W%4==0, OUT_W_PAD%(8|4)==0, STEP_H==1|2, STEP_W==1|2, GROUP==1, 
 * ConvHx4ReluStream<58,16,5,8,2,2,1,1,64,10,4,1,1> total = 313712
 * ConvHx4ReluStream<26,28,24,24,1,1,1,2,4,3,3,1,1> total = 14847
 * ConvHx4ReluStream<26,28,24,24,1,1,1,2,4,3,3,2,1> total = 8857
 * ConvHx4ReluStream<24,24,11,12,2,2,1,2,4,3,3,1,1> total = 7012
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvHx4ReluStream {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP; // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1,C_PER_M,H,W)
    static constexpr int CKK_ROW_SIZE = C_PER_M*(KH*KW+3)/4*4;

    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[CKK_ROW_SIZE];

  public:
    ConvHx4ReluStream(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(KW<=4);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0 && STEP_W==1 || OUT_W_PAD%4==0 && STEP_W==2);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
      REGISTER_FUNCTION(ConvHx4ReluStream::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires KH==KW==3, INP_W%4==0, OUT_W_PAD%8==0, STEP_H==1, STEP_W==1, GROUP==1, 
 * ConvHx4ReluStreamMultiRow<26,28,24,24,1,1,1,2,4,3,3,1,1> total = 15137
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvHx4ReluStreamMultiRow {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int CKK_ROW_SIZE = C*(KH*KW+3)/4*4;
    
    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[CKK_ROW_SIZE];
    alignas(32) float out_row[OUT_W_PAD];

  public:
    ConvHx4ReluStreamMultiRow(
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
      REGISTER_FUNCTION(ConvHx4ReluStreamMultiRow::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for OUT_W == 4 < 8, stores biases,
 * requires KW<=3, INP_W%4==0, OUT_W_PAD==4, STEP_H==1, STEP_W==1, 
 * 
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class ConvHx4Out4ReluStream {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP; // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1,C_PER_M,H,W)
    static constexpr int CKK_ROW_SIZE = C_PER_M*(KH*KW+3)/4*4;

    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[CKK_ROW_SIZE];

  public:
    ConvHx4Out4ReluStream(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(KW<=4);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD == 4);
      static_assert(STEP_H == 1);
      static_assert(STEP_W == 1);
      REGISTER_FUNCTION(ConvHx4Out4ReluStream::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for BCHW, stores biases,
 * requires KH==KW==1, INP_W%4==0, OUT_W_PAD%(8|4)==0, STEP_H==1|2, STEP_W==1|2, GROUP==1, 
 * Conv1x1ReluStream<24,24,24,24,1,1,1,2,4,1,1,1,1> total = 4867
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv1x1ReluStream {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int CKK_ROW_SIZE = (C+3)/4*4;

    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[CKK_ROW_SIZE];

  public:
    Conv1x1ReluStream(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(GROUP == 1);
      static_assert(KH==1);
      static_assert(KW==1);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD%8==0 && STEP_W==1 || OUT_W_PAD%4==0 && STEP_W==2);
      static_assert(STEP_H == 1 || STEP_H == 2);
      static_assert(STEP_W == 1 || STEP_W == 2);
      REGISTER_FUNCTION(Conv1x1ReluStream::filter);
      REGISTER_PARAMETER(bias);
    }

};


/**
 * @brief Vector stream implementation for OUT_W == 4 < 8, stores biases,
 * requires KH==KW==1, INP_W%4==0, OUT_W_PAD==4, STEP_H==1, STEP_W==1,
 * 
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
class Conv1x1Out4ReluStream {

  private:
    static constexpr int OUT_H = (INP_H - KH) / STEP_H + 1;
    static constexpr int C_PER_M = C / GROUP; // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1,C_PER_M,H,W)
    static constexpr int CKK_ROW_SIZE = (C_PER_M+3)/4*4;

    alignas(32) float (&bias)[M];
    alignas(32) float ckk_row[CKK_ROW_SIZE];

  public:
    Conv1x1Out4ReluStream(
      float (&b)[M]
    ): bias(b) {}; 

    void filter(
      input_window<float>* in,      // BCHW
      input_stream<float>* weights, // MCKK
      output_stream<float>* out     // BMHW
    );
    
    static void registerKernelClass() {
      static_assert(KW<=4);
      static_assert(INP_W%4==0);
      static_assert(OUT_W_PAD == 4);
      static_assert(STEP_H == 1);
      static_assert(STEP_W == 1);
      REGISTER_FUNCTION(Conv1x1Out4ReluStream::filter);
      REGISTER_PARAMETER(bias);
    }

};
/** @}*/


#endif // CONV_H_
