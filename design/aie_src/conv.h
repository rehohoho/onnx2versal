#ifndef CONV_H_
#define CONV_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalar {

  private:
    alignas(32) float (&weights)[M*K*K*C];
    alignas(32) float (&bias)[M];

  public:
    ConvReluScalar(
      float (&w)[M*K*K*C], // only accepts reference to MKKC array
      float (&b)[M]
    ): weights(w), bias(b) {}; 

    void filter(
      input_window<float>* in,      // BHWC (1x28x28x1)
      output_window<float>* out     // BHWM (1x24x24x6)
    );
    
    static void registerKernelClass() {
      REGISTER_FUNCTION(ConvReluScalar::filter);
      REGISTER_PARAMETER(weights);
      REGISTER_PARAMETER(bias);
    }

};

#endif // CONV_H_
