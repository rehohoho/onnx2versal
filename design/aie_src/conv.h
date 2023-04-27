#ifndef CONV_H_
#define CONV_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalar {

  private:
    alignas(32) float weights[M*K*K*C];
    alignas(32) float bias[M];

  public:
    ConvReluScalar(
      const float (&w)[M*K*K*C], // only accepts reference to MKKC array
      const float (&b)[M]
    ) {
      for (int i = 0; i < M*K*K*C; i++)
        weights[i] = w[i];
      for (int i = 0; i < M; i++)
        bias[i] = b[i];
    };

    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalar::filter);
    }

};

#endif // CONV_H_
