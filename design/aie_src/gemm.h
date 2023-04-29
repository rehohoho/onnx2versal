#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>
#include "aie_api/aie.hpp"


// xA^T + b as per torch,nn.Linear
template <int M, int K, int NCHUNK>
class GemmReluScalar {
  
  private:
    alignas(32) float (&weights)[NCHUNK*K]; // NCHUNKxK (120x256)
    alignas(32) float (&bias)[NCHUNK];      // NCHUNK   (120/?)
  
  public:
    GemmReluScalar (
      float (&w)[NCHUNK*K],
      float (&b)[NCHUNK]
    ): weights(w), bias(b) {};

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120/?)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    };
};

#endif // GEMM_H_
