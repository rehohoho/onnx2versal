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
template <int M, int K, int NCHUNK>
class GemmReluScalarGmemParamMKNK {
  public:
    void filter(
      input_window<float>* in,      // MxK  (1x256)
      input_window<float>* weight,  // NxK  (120x256)
      input_window<float>* bias,    // N    (120)
      output_window<float>* out     // MxN  (1x120)
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
template <int M, int K, int NCHUNK>
class GemmReluScalarMKNK {
  
  private:
    alignas(32) float (&weights)[NCHUNK*K]; // NCHUNKxK (120x256)
    alignas(32) float (&bias)[NCHUNK];      // NCHUNK   (120)
  
  public:
    GemmReluScalarMKNK (
      float (&w)[NCHUNK*K],
      float (&b)[NCHUNK]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120)
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
template <int M, int K, int NCHUNK>
class GemmReluScalarMKKN {
  
  private:
    static const int NCHUNK_RND = (NCHUNK + 3)/4*4;
    alignas(32) float (&weights)[NCHUNK_RND*K]; // KxNCHUNK (256x120)
    alignas(32) float (&bias)[NCHUNK];          // NCHUNK   (120)
  
  public:
    GemmReluScalarMKKN (
      float (&w)[NCHUNK_RND*K],
      float (&b)[NCHUNK]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};



/**
 * @brief Vector implementation for MK*KN, stores weights and biases, requires K%2=0, NCHUNK%4=0
 * GemmReluMKKN<1, 86, 10> total = 366
 */
template <int M, int K, int NCHUNK>
class GemmReluMKKN {
  
  private:
    static const int NCHUNK_RND = (NCHUNK + 3)/4*4;
    alignas(32) float (&weights)[NCHUNK_RND*K]; // KxNCHUNK (256x120)
    alignas(32) float (&bias)[NCHUNK];          // NCHUNK   (120)
  
  public:
    GemmReluMKKN (
      float (&w)[NCHUNK_RND*K],
      float (&b)[NCHUNK]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120)
    );
    
    static void registerKernelClass() {
      assert(K%2==0 && NCHUNK%4==0);
      REGISTER_FUNCTION(GemmReluMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};
/** @}*/


#endif // GEMM_H_
