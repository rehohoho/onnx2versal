#include <limits>

#include "pool.h"
#include "kernel_utils.h"


/*
Assumes INP_W divisible by K
*/
template <int INP_W, int OUT_W, int B, int C>
void MaxpoolScalarBHWC<INP_W, OUT_W, B, C>::filter(
  input_window<float>* in,      // BHWC (1x24x24x6)
  output_window<float>* out     // BPQC (1x12x12x6)
) {
  PROFILE_HEADER;
  printf("Running MaxpoolScalarBHWC::filter<%d, %d, %d, %d>\n", INP_W, OUT_W, B, C);

  const int K = INP_W / OUT_W;

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < OUT_W; h++) {
      for (int w = 0; w < OUT_W; w++) {

        float arr[C] = {-std::numeric_limits<double>::infinity()};
        for (int p = 0; p < K; p++) {
          for (int q = 0; q < K; q++) {
            for (int c = 0; c < C; c++) {
              float a = window_readincr(in);
              // if (c == 0) printf("%f ", a);
              arr[c] = (arr[c] < a) ? a : arr[c];
            }
          }
          window_incr(in, C*(-K+INP_W)); // go back K, go down 1
          // printf("\n");
        }
        
        for (int c = 0; c < C; c++)
          window_writeincr(out, arr[c]);

        window_incr(in, C*(-K*INP_W + K)); // go up K, go right K (next pos)
      }
      window_incr(in, C*(K-1)*INP_W); // go down K-1
    }
  }

  PROFILE_FOOTER;
}


/*
Assumes INP_W divisible by K
*/
template <int INP_W, int OUT_W, int B, int C>
void MaxpoolScalarBCHW<INP_W, OUT_W, B, C>::filter(
  input_window<float>* in,      // BCHW (1x6x24x24)
  output_window<float>* out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER;
  printf("Running MaxpoolScalarBCHW::filter<%d, %d, %d, %d>\n", INP_W, OUT_W, B, C);

  const int K = INP_W / OUT_W;

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < OUT_W; h++) {
        for (int w = 0; w < OUT_W; w++) {

          float c = -std::numeric_limits<double>::infinity();
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              float a = window_readincr(in);
              c = (a > c) ? a : c;
            }
            window_incr(in, -K+INP_W); // left K, down 1
          }
          window_incr(in, -K*INP_W + K); // up K, right K
          window_writeincr(out, c);

        } // W
        window_incr(in, -INP_W + K*INP_W); // left INP_W, down K
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}
