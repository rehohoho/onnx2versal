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
 * GemmReluScalarGmemParamMKNK<1, 86, 10> total = 19223
 */
template <int M, int K, int N, int _unused_NPAD>
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
 * @brief Scalar implementation for MK*NK, stores weights and biases,
 * Running GemmReluScalarMKNK<1, 86, 10> total = 1939
 */
// xA^T + b as per torch,nn.Linear
template <int M, int K, int N, int _unused_NPAD>
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
 * GemmReluScalarMKKN<1, 86, 10> total = 1939
 */
template <int M, int K, int N, int NPAD>
class GemmReluScalarMKKN {
  
  private:
    alignas(32) float (&weights)[K*NPAD]; // KxN
    alignas(32) float (&bias)[N];         // N
  
  public:
    GemmReluScalarMKKN (
      float (&w)[NPAD*K],
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
 * @brief Vector implementation for MK*KN, stores weights and biases, requires K%2=0, N%4=0
 * GemmReluMKKN<1, 86, 10> total = 366
 */
template <int M, int K, int N, int NPAD>
class GemmReluMKKN {
  
  private:
    alignas(32) float (&weights)[K*NPAD]; // KxN
    alignas(32) float (&bias)[N];         // N

    static constexpr int K_REM8 = K%8;
    static constexpr int RUN_LASTCHUNK = K_REM8 > 0;
  
  public:
    GemmReluMKKN (
      float (&w)[NPAD*K],
      float (&b)[N]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,  // MxK
      output_window<float>* out // MxN
    );
    
    static void registerKernelClass() {
      assert(K%2==0 && NPAD%4==0);
      REGISTER_FUNCTION(GemmReluMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};
/** @}*/


#endif // GEMM_H_
