#include "qlinearconv.h"
#include "kernel_utils.h"


template <int INP_W, int OUT_W, int B, int C, int M, int K>
void QLinearConvScalar<INP_W, OUT_W, B, C, M, K>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QLinearConvScalar<%d, %d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M, K));

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      for (int h = 0; h < OUT_W; h++) {
        for (int w = 0; w < OUT_W; w++) {
        
          // qy = qy_zero + [(qx-qx_zero)*(qw-qw_zero) + qbias] * qx_scale*qw_scale/qy_scale
          int res = bias[m];
          weightIdx = m*C*K*K;
          
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              for (int q = 0; q < K; q++) {
                int a = window_readincr(in) - x_zero_point;
                res += a * (weights[weightIdx]-w_zero_point);
                weightIdx++;
              }
              window_incr(in, -K+INP_W); // go left K, down 1
            }
            window_incr(in, -K*INP_W + INP_W*INP_W); // go up K, channel 1
          }
          res = y_zero_point + round(x_scale*w_scale/y_scale * res);
          
          // saturate at the end only
          res = std::min(std::max(res, -128), 128);

          // if (res < 0) res = 0;
          window_writeincr(out, res);
          window_incr(in, -C*INP_W*INP_W + 1); // go channel -C, right 1
        }

        window_incr(in, INP_W-OUT_W); // go left OUT_W, go down 1
      }
      window_incr(in, -OUT_W*INP_W); // go up OUT_W
    }
  }

  PROFILE_FOOTER;
}
