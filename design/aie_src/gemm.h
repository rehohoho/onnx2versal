#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>


// xA^T + b as per torch,nn.Linear
template <int M, int K, int NCHUNK>
class GemmReluScalarMKNK {
  
  private:
    alignas(32) float (&weights)[NCHUNK*K]; // NCHUNKxK (120x256)
    alignas(32) float (&bias)[NCHUNK];      // NCHUNK   (120/?)
  
  public:
    GemmReluScalarMKNK (
      float (&w)[NCHUNK*K],
      float (&b)[NCHUNK]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120/?)
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
    alignas(32) float (&weights)[NCHUNK*K]; // KxNCHUNK (256x120)
    alignas(32) float (&bias)[NCHUNK];      // NCHUNK   (120/?)
  
  public:
    GemmReluScalarMKKN (
      float (&w)[NCHUNK*K],
      float (&b)[NCHUNK]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120/?)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalarMKKN::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};


#endif // GEMM_H_
