#include "conv.h"
#include "kernel_utils.h"


template <int INP_W, int OUT_W, int B, int C, int M, int K>
void conv_relu_scalar(
	input_window<float>* in,      // BHWC (1x28x28x1)
  input_window<float>* weight,  // MKKC (6x5x5x1)
  input_window<float>* bias,    // M    (6)
  output_window<float>* out     // BHWM (1x24x24x6)
) {

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < OUT_W; h++) {
      for (int w = 0; w < OUT_W; w++) {
        
        for (int m = 0; m < M; m++) { 

          // KKC
          float res = window_readincr(bias);
          
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              for (int c = 0; c < C; c++) {
                float a = window_readincr(in);
                float b = window_readincr(weight);
                // if (m == 0) printf("%f ", a);
                res += a * b;
              }
            }
            window_incr(in, C*(-K + INP_W)); // go back K, go down 1
            // if (m == 0) printf("\n");
          }

          if (res < 0) res = 0;
          window_writeincr(out, res);
          window_incr(in, C*(-K*INP_W)); // go up K
        }

        window_incr(in, C); // next position
      }
      window_incr(in, C*K - C); // next row
    }
  }

}
