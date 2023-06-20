#include <limits>

#include "qlinearpool.h"
#include "kernel_utils.h"


#define QLINEARPOOL_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_H, OUT_W, B, C);


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
void QLinearAvgpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>::filter(
  input_window<TT>* in,      // BCHW (1x6x24x24)
  output_window<TT>* out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER2;

  float scale = in_scale * inv(K*K * out_scale);

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {

          int sum = -in_zero *K*K;
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              sum += window_readincr(in);
            }
            window_incr(in, -K+INP_W); // left K, down 1
          }
          window_incr(in, -K*INP_W + K); // up K, right K
          
          sum = round(sum * scale) + out_zero;
          sum = std::min(std::max(sum, -128), 127);

          window_writeincr(out, (int8_t) sum);
        } // W
        window_incr(in, -OUT_W*K  + K*INP_W); // down K, left OUT_W*K , account for padding
      } // H
    } // C
  } // B

  QLINEARPOOL_PROFILE_FOOTER("QLinearAvgpoolScalarBCHW");
}