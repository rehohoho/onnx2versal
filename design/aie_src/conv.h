#ifndef CONV_H_
#define CONV_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int INP_W, int OUT_W, int B, int C, int M, int K>
void conv_scalar(
	input_window<float>* in,      // BHWC (1x28x28x1)
  input_window<float>* weight,  // MKKC (6x5x5x1)
  input_window<float>* bias,    // M    (6)
  output_window<float>* out     // BHWM (1x24x24x6)
);

#endif // CONV_H_
