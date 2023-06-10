#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>
#include <assert.h>


/** 
 * @defgroup GemmKernels
 * @ingroup Gemm
 * 
 * @details 
 * Design notes
 * - Single acc/fpmac: 393 cycles, no fpmac interleaving
 * - upd_w v8 better than load v16 from pointer, allows interleaving: 406 -> 698
 * - interleaving ops: 406 -> 365
 * - kstep=4 to 8: 365 -> 339
 * - Typically K > N for downsampling, M=1 if each net does an instance
 * - If chunking by N, for N%16=0, K<=128, for N%8=0, K<=256
 * 
 * @{
 */


/**
 * @brief Scalar implementation for MK*NK, streams input, outputs, weights, stores bias 
 * GemmReluScalarMKNKStream<7,36,10> total = 12214
 */
template <int M, int K, int N, int IS_RELU>
class GemmReluScalarMKNKStream {
  private:
    alignas(32) float (&bias)[N];
    alignas(32) float in_row[K];

  public:
    GemmReluScalarMKNKStream (
      float (&b)[N]
    ): bias(b) {};

    void filter(
      input_stream<float>* in,      // MxK
      input_stream<float>* weight,  // NxK
      output_window<float>* out     // MxN
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarMKNKStream::filter);
      REGISTER_PARAMETER(bias);
    };
};


/**
 * @brief Scalar implementation for MK*NK, stores weights and biases,
 * Running GemmReluScalarMKNK<7,36,10> total = 6515
 */
// xA^T + b as per torch,nn.Linear
template <int M, int K, int N, int IS_RELU>
class GemmReluScalarMKNK {
  
  private:
    alignas(32) float (&weights)[N*K]; // NxK
    alignas(32) float (&bias)[N];      // N
  
  public:
    GemmReluScalarMKNK (
      float (&w)[N*K],
      float (&b)[N]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,  // MxK
      output_window<float>* out // MxN
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarMKNK::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};


/**
 * @brief Scalar implementation for MK*KN, stores weights and biases,
 * GemmReluScalarMKKN<7,36,10> total = 7798
 */
template <int M, int K, int N, int IS_RELU>
class GemmReluScalarMKKN {
  
  private:
    alignas(32) float (&weights)[K*N]; // KxN
    alignas(32) float (&bias)[N];      // N
  
  public:
    GemmReluScalarMKKN (
      float (&w)[N*K],
      float (&b)[N]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,  // MxK
      output_window<float>* out // MxN
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};


/**
 * @brief Vector implementation for MK*KN, streams input, outputs, weights, stores bias,
 * requires K%4==0 and N%8==0
 * GemmReluMKKNStream<7,36,10> total = 3630
 */
template <int M, int K, int N, int IS_RELU>
class GemmReluMKKNStream {
  private:
    alignas(32) float (&bias)[N];
    alignas(32) float in_row[4*K];
    alignas(32) float out_row[3*N];

  public:
    GemmReluMKKNStream (
      float (&b)[N]
    ): bias(b) {};

    void filter(
      input_stream<float>* in,      // MxK
      input_stream<float>* weight,  // NxK
      output_window<float>* out     // MxN
    );
    static void registerKernelClass() {
      static_assert(K%4==0 && N%8==0 && (4*K + 3*N)*4 <= 16384);
      REGISTER_FUNCTION(GemmReluMKKNStream::filter);
      REGISTER_PARAMETER(bias);
    };
};


/**
 * @brief Vector implementation for MK*KN, stores weights and biases, 
 * requires K%4=0, N%4=0
 * GemmReluMKKN<7,36,10> total = 1470
 */
template <int M, int K, int N, int IS_RELU>
class GemmReluMKKN {
  
  private:
    alignas(32) float (&weights)[K*N]; // KxN
    alignas(32) float (&bias)[N];      // N

    static constexpr int K_REM8 = K%8;
    static constexpr int RUN_LASTCHUNK = K_REM8 > 0;
  
  public:
    GemmReluMKKN (
      float (&w)[N*K],
      float (&b)[N]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,  // MxK
      output_window<float>* out // MxN
    );
    
    static void registerKernelClass() {
      static_assert(K%4==0 && N%4==0);
      REGISTER_FUNCTION(GemmReluMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};
/** @}*/


#endif // GEMM_H_
