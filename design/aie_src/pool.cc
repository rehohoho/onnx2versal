#include <limits>

#include "pool.h"
#include "kernel_utils.h"


template <typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int B, int C>
void MaxpoolScalarBHWC<TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C>::filter(
  input_window<TT>* in,      // BHWC (1x24x24x6)
  output_window<TT>* out     // BPQC (1x12x12x6)
) {
  PROFILE_HEADER(printf(
    "Running MaxpoolScalarBHWC::filter<%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C));

  const int K = INP_W / OUT_W;
  const TT min = std::numeric_limits<TT>::lowest();

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < INP_H/K; h++) {
      for (int w = 0; w < OUT_W; w++) {

        TT arr[C] = {min};
        for (int p = 0; p < K; p++) {
          for (int q = 0; q < K; q++) {
            for (int c = 0; c < C; c++) {
              TT a = window_readincr(in);
              arr[c] = (arr[c] < a) ? a : arr[c];
            }
          }
          window_incr(in, C*(-K+INP_W_PAD)); // go back K, go down 1
        }
        
        for (int c = 0; c < C; c++)
          window_writeincr(out, arr[c]);

        window_incr(in, C*(-K*INP_W_PAD + K)); // go up K, go right K (next pos)
      }
      window_incr(out, OUT_W_PAD - OUT_W);
      window_incr(in, C*(-INP_W + K*INP_W_PAD)); // go down K, go left INP_W, account for padding
    }
  }

  PROFILE_FOOTER;
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int B, int C>
void MaxpoolScalarBCHW<TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C>::filter(
  input_window<TT>* in,      // BCHW (1x6x24x24)
  output_window<TT>* out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER(printf(
    "Running MaxpoolScalarBHWC::filter<%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C));

  const int K = INP_W / OUT_W;
  const TT min = std::numeric_limits<TT>::lowest();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < INP_H/K; h++) {
        for (int w = 0; w < OUT_W; w++) {

          TT c = min;
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              TT a = window_readincr(in);
              printf("%d ", a);
              c = (a > c) ? a : c;
            }
            window_incr(in, -K+INP_W_PAD); // left K, down 1
          }
          window_incr(in, -K*INP_W_PAD + K); // up K, right K
          window_writeincr(out, c);
          printf("%d | ", c);
        } // W
        printf("\n");
        window_incr(out, OUT_W_PAD - OUT_W);
        window_incr(in, -INP_W + K*INP_W_PAD); // down K, left INP_W, account for padding
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int B, int C>
void Maxpool2x2BCHW<TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C>::filter(
  input_window<float>* in_window,      // BCHW (1x6x24x24)
  output_window<float>* out_window     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER(printf(
    "Running Maxpool2x2BCHW::filter<%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, B, C));

  const int K = INP_W / OUT_W;
  const float min = std::numeric_limits<float>::lowest();

  v8float *in0 = (v8float *) in_window->ptr + 0 * INP_W_PAD/8;
  v8float *in1 = (v8float *) in_window->ptr + 1 * INP_W_PAD/8;
  v8float *in2 = (v8float *) in_window->ptr + 2 * INP_W_PAD/8;
  v8float *in3 = (v8float *) in_window->ptr + 3 * INP_W_PAD/8;
  v16float v = null_v16float();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < INP_H; h+=4) {
        for (int w = 0; w < INP_W; w+=8) {  // computes 2x4 cells with 4x8 cells

          v8float res = aie::broadcast<float, 8>(min);
          v = upd_w(v, 0, *in0);
          v = upd_w(v, 1, *in2);
          res = fpmax(res, v, 0, 0xeca86420);
          res = fpmax(res, v, 0, 0xfdb97531);
          
          v = upd_w(v, 0, *in1);
          v = upd_w(v, 1, *in3);
          res = fpmax(res, v, 0, 0xeca86420);
          res = fpmax(res, v, 0, 0xfdb97531);

          window_write(out_window, ext_v(res, 0));
          window_incr(out_window, OUT_W_PAD);
          window_write(out_window, ext_v(res, 1));
          window_incr(out_window, -OUT_W_PAD+4);
          
          in0++;
          in1++;
          in2++;
          in3++;
        } // W
        in0 += 4*INP_W_PAD/8 - INP_W/8; // account for padding
        in1 += 4*INP_W_PAD/8 - INP_W/8;
        in2 += 4*INP_W_PAD/8 - INP_W/8;
        in3 += 4*INP_W_PAD/8 - INP_W/8;
        window_incr(out_window, OUT_W_PAD - OUT_W);
        window_incr(out_window, OUT_W_PAD);
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}