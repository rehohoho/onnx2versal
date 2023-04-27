#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>
#include "aie_api/aie.hpp"


// xA^T + b as per torch,nn.Linear
template <int M, int K, int N>
class GemmReluScalar {
  
  private:
    alignas(32) float weights[N*K]; // NxK (120x256)
    alignas(32) float bias[N];      // N   (120)
  
  public:
    GemmReluScalar (
      const float (&w)[N*K],
      const float (&b)[N]
    ) {
      for (int i = 0; i < N*K; i++)
        weights[i] = w[i];
      for (int i = 0; i < N; i++)
        bias[i] = b[i];
    };

    void filter(
      input_window<float>* in,      // MxK  (1x256)
      output_window<float>* out     // MxN  (1x120)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(GemmReluScalar::filter);
    };
};

#endif // GEMM_H_
