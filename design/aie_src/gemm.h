#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>
#include "aie_api/aie.hpp"


// xA^T + b as per torch,nn.Linear
template <int M, int K, int NCHUNK>
class GemmReluScalar {
  
  private:
    alignas(32) float weights[NCHUNK*K]; // NCHUNKxK (120x256)
    alignas(32) float bias[NCHUNK];      // NCHUNK   (120/?)
    int nOff;
  
  public:
    GemmReluScalar (
      const float (&w)[NCHUNK*K],
      const float (&b)[NCHUNK],
      int nOff
    ) {
      for (int i = 0; i < NCHUNK*K; i++)
        weights[i] = w[i];
      for (int i = 0; i < NCHUNK; i++)
        bias[i] = b[i];
      this->nOff = nOff;
    };

    void filter(
      input_window<float>* in,      // MxK       (1x256)
      output_window<float>* out     // MxNCHUNK  (1x120/?)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalar::filter);
    };
};

#endif // GEMM_H_
