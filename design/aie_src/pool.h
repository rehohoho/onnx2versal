#ifndef POOL_H_
#define POOL_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int INP_W, int OUT_W, int B, int C>
class MaxpoolScalarBHWC {
  public:
    void filter(
      input_window<float>* in,      // BHWC (1x24x24x6)
      output_window<float>* out     // BPQC (1x12x12x6)
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBHWC::filter);
    }
};


template <int INP_W, int OUT_W, int B, int C>
class MaxpoolScalarBCHW {
  public:
    void filter(
      input_window<float>* in,      // BCHW (1x6x24x24)
      output_window<float>* out     // BCPQ (1x6x12x12)
    );

    static void registerKernelClass() {
      REGISTER_FUNCTION(MaxpoolScalarBCHW::filter);
    }
};


#endif // POOL_H_
