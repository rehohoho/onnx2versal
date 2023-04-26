#ifndef POOL_H_
#define POOL_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int INP_W, int OUT_W, int B, int C>
void maxpool_scalar(
	input_window<float>* in,      // BHWC (1x28x28x1)
  output_window<float>* out     // BPQC (1x24x24x6)
);

#endif // POOL_H_
