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
 * @brief Scalar implementation for MK*NK, streams weights and biases, 
 * GemmReluScalarGmemParamMKNK<2, 36, 10> total = 16449
 */
template <int M, int K, int N, int IS_RELU>
class GemmReluScalarGmemParamMKNK {
  public:
    void filter(
      input_window<float>* in,      // MxK
      input_window<float>* weight,  // NxK
      input_window<float>* bias,    // N 
      output_window<float>* out     // MxN
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarGmemParamMKNK::filter);
    };
};


/**
 * @brief Scalar implementation for MK*KN, streams weights and biases, 
 * GemmReluScalarGmemParamMKNK<2, 36, 10> total = 16590
 */
template <int M, int K, int N, int IS_RELU>
class GemmReluScalarGmemParamMKKN {
  public:
    void filter(
      input_window<float>* in,      // MxK
      input_window<float>* weight,  // NxK
      input_window<float>* bias,    // N 
      output_window<float>* out     // MxN
    );
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarGmemParamMKKN::filter);
    };
};


/**
 * @brief Scalar implementation for MK*NK, stores weights and biases,
 * Running GemmReluScalarMKNK<2, 36, 10> total = 1874
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
 * GemmReluScalarMKKN<2, 36, 10> total = 2242
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
 * @brief Vector implementation for MK*KN, stores weights and biases, 
 * requires K%4=0, N%4=0
 * GemmReluMKKN<2, 36, 10> total = 441
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
