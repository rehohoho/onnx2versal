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
  PROFILE_HEADER(printf(
    "Running MaxpoolScalarBHWC::filter<%d, %d, %d, %d>\n", INP_W, OUT_W, B, C));

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
  PROFILE_HEADER(printf(
    "Running MaxpoolScalarBCHW::filter<%d, %d, %d, %d>\n", INP_W, OUT_W, B, C));

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


/*
Assumes INP_W divisible by K
Assumes OUT_W divisible by 8

Only up to 64 floats in vector registers
1) 2x v16float, 1x fpmax
2) 4x v8float, 2x fpmax

Definitely bandwidth limited
- 2 accs will blow the 64-float vector regs (901 -> 1330 cycles)
- Using window_incr instead of pointers (901 -> 988 cycles)
- Concat adds additional computation (901 -> 1468 cycles)
*/
template <int INP_W, int OUT_W, int B, int C>
void Maxpool2x2BCHW<INP_W, OUT_W, B, C>::filter(
  input_window<float>* in_window,      // BCHW (1x6x24x24)
  output_window<float>* out_window     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER(printf(
    "Running Maxpool2x2BCHW::filter<%d, %d, %d, %d>\n", INP_W, OUT_W, B, C));

  const int K = INP_W / OUT_W;
  v8float *in0 = (v8float *) in_window->ptr + 0 * INP_W/8;
  v8float *in1 = (v8float *) in_window->ptr + 1 * INP_W/8;
  v8float *in2 = (v8float *) in_window->ptr + 2 * INP_W/8;
  v8float *in3 = (v8float *) in_window->ptr + 3 * INP_W/8;
  v16float v = null_v16float();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < INP_W; h+=4) {
        for (int w = 0; w < INP_W; w+=8) {  // computes 2x4 cells with 4x8 cells

          v8float res = aie::broadcast<float, 8>(-std::numeric_limits<float>::infinity());
          v = upd_w(v, 0, *in0);
          v = upd_w(v, 1, *in2);
          res = fpmax(res, v, 0, 0xeca86420);
          res = fpmax(res, v, 0, 0xfdb97531);
          
          v = upd_w(v, 0, *in1);
          v = upd_w(v, 1, *in3);
          res = fpmax(res, v, 0, 0xeca86420);
          res = fpmax(res, v, 0, 0xfdb97531);

          window_write(out_window, ext_v(res, 0));
          window_incr(out_window, OUT_W);
          window_write(out_window, ext_v(res, 1));
          window_incr(out_window, -OUT_W+4);
          
          in0++;
          in1++;
          in2++;
          in3++;
        } // W
        in0 += 3*INP_W/8;
        in1 += 3*INP_W/8;
        in2 += 3*INP_W/8;
        in3 += 3*INP_W/8;
        window_incr(out_window, OUT_W);
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}