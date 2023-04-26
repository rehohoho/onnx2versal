#ifndef GEMM_H_
#define GEMM_H_

#include <adf.h>
#include "aie_api/aie.hpp"


// xA^T + b as per torch,nn.Linear
template <int M, int K, int N>
void gemm_relu_scalar(
	input_window<float>* in,      // MxK  (1x256)
  input_window<float>* weight,  // NxK  (120x256)
  input_window<float>* bias,    // N    (120)
  output_window<float>* out     // MxN  (1x120)
);

#endif // GEMM_H_
