#include <limits>

#include "qlinearpool.h"
#include "kernel_utils.h"


#define QLINEARPOOL_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_H, OUT_W, B, C);


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
void QLinearAvgpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>::filter(
  input_window<TT>* in,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  int resvi = 0;
  v16int16 resv = null_v16int16();

#define WRITE_OUT(res) \
  resv = upd_elem(resv, resvi, res); \
  if (resvi == 15) writeincr_v16(out, pack(resv)); \
  resvi = (resvi + 1) & 0xf;

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

          WRITE_OUT(sum);
        } // W
        window_incr(in, -OUT_W*K  + K*INP_W); // down K, left OUT_W*K , account for padding
      } // H
    } // C
  } // B
#undef WRITE_OUT

  QLINEARPOOL_PROFILE_FOOTER("QLinearAvgpoolScalarBCHW");
}