#ifndef CONV_H_
#define CONV_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int B, int H, int W, int C>
void bhwc2bchw_scalar(
	input_window<float>* in,      // BHWC (1x4x4x16)
  output_window<float>* out     // BCHW (1x16x4x4)
);

#endif // CONV_H_
