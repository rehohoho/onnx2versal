#include "gemm.h"
#include "kernel_utils.h"


// xA^T + b as per torch,nn.Linear
template <int M, int K, int N>
void gemm_relu_scalar(
	input_window<float>* in,      // MxK  (1x256)
  input_window<float>* weight,  // NxK  (120x256)
  input_window<float>* bias,    // N    (120)
  output_window<float>* out     // MxN  (1x120)
) {
  PROFILE_HEADER;
  printf("Running gemm_relu_scalar<%d, %d, %d>", M, K, N);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float res = window_readincr(bias);
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = window_readincr(weight);
        res += a * b; // matB is a circular buffer
      }    
      
      if (res < 0) res = 0;
      window_writeincr(out, res);
    }
    window_incr(in, K); // next row
    window_incr(out, 1); // next row
  }

  PROFILE_FOOTER;
}
