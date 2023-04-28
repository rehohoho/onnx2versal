#include "gemm.h"
#include "kernel_utils.h"


/*
xA^T + b as per torch,nn.Linear
1956 cycles for lenet fc3
MxK * NxK
weights[N*K] (120x256)
bias[N]      (120)
*/
template <int M, int K, int N>
void GemmReluScalar<M, K, N>::filter(
	input_window<float>* in,      // MxK  (1x256)
  output_window<float>* out     // MxN  (1x120)
) {
  PROFILE_HEADER;
  printf("Running gemm_relu_scalar<%d, %d, %d>", M, K, N);
  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    window_incr(out, nOff);  // move to column offset for each i-th out row
    for (int j = 0; j < N; j++) {
      float res = bias[j];
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = weights[weightIdx];
        weightIdx++;
        res += a * b;
      }    
      
      if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}
