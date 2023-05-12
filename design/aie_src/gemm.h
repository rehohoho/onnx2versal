#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>


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


/*
Expects N%4=0, K%4=0, pad accordingly
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
      REGISTER_FUNCTION(GemmReluMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};


#endif // GEMM_H_
