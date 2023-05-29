#include <limits>

#include "pool.h"
#include "kernel_utils.h"


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
void MaxpoolScalarBHWC<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>::filter(
  input_window<TT>* in,      // BHWC (1x24x24x6)
  output_window<TT>* out     // BPQC (1x12x12x6)
) {
  PROFILE_HEADER(printf(
    "Running MaxpoolScalarBHWC::filter<%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C));

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
          window_incr(in, C*(-K+INP_W)); // go back K, go down 1
        }
        
        for (int c = 0; c < C; c++)
          window_writeincr(out, arr[c]);

        window_incr(in, C*(-K*INP_W + K)); // go up K, go right K (next pos)
      }
      window_incr(in, C*(-INP_W + K*INP_W)); // go down K, go left INP_W, account for padding
    }
  }

  PROFILE_FOOTER;
}


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
void MaxpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>::filter(
  input_window<TT>* in,      // BCHW (1x6x24x24)
  output_window<TT>* out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER(printf(
    "Running MaxpoolScalarBCHW::filter<%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C));

  const TT min = std::numeric_limits<TT>::lowest();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < INP_H/K; h++) {
        for (int w = 0; w < OUT_W; w++) {

          TT c = min;
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              TT a = window_readincr(in);
              c = (a > c) ? a : c;
            }
            window_incr(in, -K+INP_W); // left K, down 1
          }
          window_incr(in, -K*INP_W + K); // up K, right K
          window_writeincr(out, c);
        } // W
        window_incr(in, -INP_W + K*INP_W); // down K, left INP_W, account for padding
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
void Maxpool2x2FloatBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>::filter(
  input_window<float>* in_window,      // BCHW (1x6x24x24)
  output_window<float>* out_window     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER(printf(
    "Running Maxpool2x2FloatBCHW::filter<%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C));

  const float min = std::numeric_limits<float>::lowest();

  v8float *in0 = (v8float *) in_window->ptr + 0 * INP_W/8;
  v8float *in1 = (v8float *) in_window->ptr + 1 * INP_W/8;
  v8float *in2 = (v8float *) in_window->ptr + 2 * INP_W/8;
  v8float *in3 = (v8float *) in_window->ptr + 3 * INP_W/8;
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
          window_incr(out_window, OUT_W);
          window_write(out_window, ext_v(res, 1));
          window_incr(out_window, -OUT_W+4);
          
          in0++;
          in1++;
          in2++;
          in3++;
        } // W
        in0 += 4*INP_W/8 - INP_W/8; // account for padding
        in1 += 4*INP_W/8 - INP_W/8;
        in2 += 4*INP_W/8 - INP_W/8;
        in3 += 4*INP_W/8 - INP_W/8;
        window_incr(out_window, OUT_W);
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}


/**
 * max32 (v64int16 xbuff, 
 *  int xstart, unsigned int xoffsets, unsigned int xoffsets_hi, unsigned int xsquare, 
 *  int ystart, unsigned int yoffsets, unsigned int yoffsets_hi, unsigned int ysquare)
 * 
 * 0x06...00, 0x0e...08 => 0 1 2 3 ... 12 13 14 15, 16 17 18 19 ... 28 29 30 31
 * max32(v, 0, 0x06040200, 0x0e0c0a08, 0x3210, 32, 0x06040200, 0x0e0c0a08, 0x3210); // first 32 with next 32
 * problem: offsets index <= 32, each 4b selects 2 adjacent lanes
 * 
 * 128 int16 max
 */
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C>
void Maxpool2x2Int8BCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C>::filter(
  input_window<int8_t>* in_window,      // BCHW (1x6x24x24)
  output_window<int8_t>* out_window     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER(printf(
    "Running Maxpool2x2Int8BCHW::filter<%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C));

  const int8_t min = std::numeric_limits<int8_t>::lowest();
  int8_t *out_ptr = (int8_t *) out_window->ptr;

  v16int8 *in0 = (v16int8 *) in_window->ptr + 0 * INP_W/16;
  v16int8 *in1 = (v16int8 *) in_window->ptr + 1 * INP_W/16;
  v64int16 v = null_v64int16();
  aie::vector<int16_t, 8> tmp = null_v8int16();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < INP_H; h+=2) {
        for (int w = 0; w <= INP_W-16; w+=32) {  // computes 1x16 cells with 2x32 cells, stop at 16 left since INP_W%16=0

          v32int16 res = aie::broadcast<int16_t, 32>(min);
          v = upd_w(v, 0, unpack(*in0)); in0++;
          v = upd_w(v, 1, unpack(*in0)); in0++;
          v = upd_w(v, 2, unpack(*in1)); in1++;
          v = upd_w(v, 3, unpack(*in1)); in1++;
          
          // first 32 against next 32: 0 1 2 3 ... 28 29 30 31 x 32 33 34 35 ... 60 61 62 63
          res = max32(v, 0, 0x06040200, 0x0e0c0a08, 0x3210, 32, 0x06040200, 0x0e0c0a08, 0x3210);
          res = shuffle32(res, 0, 0x06040200, 0x0e0c0a08, 0x3120); // 0213 4657 ... 28302931
          // alternate adjacent lanes: 0 1 4 5 ... 24 25 28 29 x 2 3 6 7 ... 26 27 30 31
          res = max32(res, 0, 0x1c181410, 0x00000000, 0x3210, 0, 0x1d191511, 0x00000000, 0x3210);

          window_writeincr(out_window, pack(ext_w(res,0)));
        } // W

        if (RUN_16CHUNK) {  // computes 1x8 cells with 2x16 cells, handle last 16
          v32int16 res = aie::broadcast<int16_t, 32>(min);
          v = upd_w(v, 0, unpack(*in0)); in0++;
          v = upd_w(v, 2, unpack(*in1)); in1++;
          res = max32(v, 0, 0x06040200, 0x0e0c0a08, 0x3210, 32, 0x06040200, 0x0e0c0a08, 0x3210);
          res = shuffle32(res, 0, 0x06040200, 0x0e0c0a08, 0x3120);
          res = max32(res, 0, 0x1c181410, 0x00000000, 0x3210, 0, 0x1d191511, 0x00000000, 0x3210);

          window_writeincr(out_window, pack(ext_w(res,0)));
        } // W
        
        in0 += 2*INP_W/16 - (INP_W+15)/16; // account for padding
        in1 += 2*INP_W/16 - (INP_W+15)/16;
      } // H
    } // C
  } // B

  PROFILE_FOOTER;
}
