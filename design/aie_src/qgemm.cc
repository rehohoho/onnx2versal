#include "qgemm.h"
#include "kernel_utils.h"


template <int M, int K, int N, int NPAD>
void QgemmScalar<M, K, N, NPAD>::filter(
	input_window<int8_t>* in,      // MxK
                                // KxNPAD
  output_window<int8_t>* out     // MxNPAD
) {
  PROFILE_HEADER(printf(
    "Running QgemmScalar<%d,%d,%d,%d>\n", M, K, N, NPAD));

  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int res = bias[j];
      weightIdx = j;
      for (int k = 0; k < K; k++) {
        int a = window_readincr(in);
        int b = weights[weightIdx];
        weightIdx += NPAD;
        res += (a-x_zero) * (b-w_zero);
      }
      res = y_zero + round(scale * res);
      res = std::min(std::max(res, -128), 127);
      window_writeincr(out, (int8_t) res);
      window_incr(in, -K); // repeat same in row for next j
    }
    for (int j = N; j < NPAD; j++)
      window_writeincr(out, y_zero);
    
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}
