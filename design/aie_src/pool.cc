#include <limits>

#include "pool.h"
#include "kernel_utils.h"


#define POOL_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W);

// MaxpoolScalarBHWC<24,24,12,12,1,6,2,2,2,2> total = 10758, with output window 7673
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
void MaxpoolScalarBHWC<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>::filter(
  input_window<TT>* in,      // BHWC (1x24x24x6)
  output_stream<TT>* restrict out     // BPQC (1x12x12x6)
) {
  PROFILE_HEADER2;

  const TT min = std::numeric_limits<TT>::lowest();

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < OUT_H; h++) {
      for (int w = 0; w < OUT_W; w++) {

        TT arr[C] = {min};
        for (int p = 0; p < KH; p++) {
          for (int q = 0; q < KW; q++) {
            for (int c = 0; c < C; c++) {
              TT a = window_readincr(in);
              arr[c] = (arr[c] < a) ? a : arr[c];
            }
          }
          window_incr(in, C*(-KW+INP_W)); // go back KW, go down 1
        }
        
        for (int c = 0; c < C; c++)
          writeincr(out, arr[c]);

        window_incr(in, C*(-KH*INP_W + STEP_W)); // go up KH, go right STEP_W (next pos)
      }
      window_incr(in, C*(-OUT_W*STEP_W + STEP_H*INP_W)); // go down STEP_H, go left OUT_W*STEP_W, account for padding
    }
  }

  POOL_PROFILE_FOOTER("MaxpoolScalarBHWC");
}


// MaxpoolScalarBCHW<24,24,12,12,1,6,2,2,2,2> total = 19174, with output_window 11302
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
void MaxpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>::filter(
  input_window<TT>* in,      // BCHW (1x6x24x24)
  output_stream<TT>* restrict out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER2;

  const TT min = std::numeric_limits<TT>::lowest();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {

          TT c = min;
          for (int p = 0; p < KH; p++) {
            for (int q = 0; q < KW; q++) {
              TT a = window_readincr(in);
              c = (a > c) ? a : c;
            }
            window_incr(in, -KW+INP_W); // left KW, down 1
          }
          window_incr(in, -KH*INP_W + STEP_W); // up KH, right STEP_W
          writeincr(out, c);
        } // W
        window_incr(in, -OUT_W*STEP_W + STEP_H*INP_W); // left OUT_W*STEP_W, down STEP_H
      } // H
      window_incr(in, (INP_H - OUT_H*STEP_H)*INP_W);
    } // C
  } // B

  POOL_PROFILE_FOOTER("MaxpoolScalarBCHW");
}


// Maxpool2x2FloatBCHW<24,24,12,12,1,6,2,2,2,2> total = 1977, with output window 901
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
void Maxpool2x2FloatBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>::filter(
  input_window<float>* in,      // BCHW (1x6x24x24)
  output_stream<float>* restrict out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER2;

  const float min = std::numeric_limits<float>::lowest();

  v8float *in0 = (v8float *) in->ptr + 0 * INP_W/8;
  v8float *in1 = (v8float *) in->ptr + 1 * INP_W/8;
  v16float v = null_v16float();

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < INP_H; h+=2) {
        for (int w = 0; w < INP_W; w+=8) {  // computes 1x4 cells with 2x8 cells

          v8float res = aie::broadcast<float, 8>(min);
          v = upd_w(v, 0, *in0);
          v = upd_w(v, 1, *in1);
          res = fpmax(res, v, 0, 0xeca86420);
          res = fpmax(res, v, 0, 0xfdb97531);
          res = fpmax(res, res, 0, 0x00007654);
          writeincr_v4(out, ext_v(res, 0));      
          in0++;
          in1++;
        } // W
        in0 += 2*INP_W/8 - INP_W/8; // account for padding
        in1 += 2*INP_W/8 - INP_W/8;
      } // H
    } // C
  } // B

  POOL_PROFILE_FOOTER("Maxpool2x2FloatBCHW");
}


// Maxpool2x2Int8BCHW<24,32,12,16,1,6,2,2,2,2> total = 973, with output window 324
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
void Maxpool2x2Int8BCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>::filter(
  input_window<TT>* in,               // BCHW (1x6x24x24)
  output_stream<TT>* restrict out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER2;

  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  const TT min = std::numeric_limits<TT>::lowest();

  v16 *in0 = (v16 *) in->ptr + 0 * INP_W/16;
  v16 *in1 = (v16 *) in->ptr + 1 * INP_W/16;
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

          writeincr_v16(out, pack(ext_w(res,0)));
        } // W

        if (RUN_16CHUNK) {  // computes 1x8 cells with 2x16 cells, handle last 16
          v32int16 res = aie::broadcast<int16_t, 32>(min);
          v = upd_w(v, 0, unpack(*in0)); in0++;
          v = upd_w(v, 2, unpack(*in1)); in1++;
          res = max32(v, 0, 0x06040200, 0x0e0c0a08, 0x3210, 32, 0x06040200, 0x0e0c0a08, 0x3210);
          res = shuffle32(res, 0, 0x06040200, 0x0e0c0a08, 0x3120);
          res = max32(res, 0, 0x1c181410, 0x00000000, 0x3210, 0, 0x1d191511, 0x00000000, 0x3210);

          writeincr_v16(out, pack(ext_w(res,0)));
        } // W
        
        in0 += 2*INP_W/16 - (INP_W+15)/16; // account for padding
        in1 += 2*INP_W/16 - (INP_W+15)/16;
      } // H
    } // C
  } // B

  POOL_PROFILE_FOOTER("Maxpool2x2Int8BCHW");
}


// AvgpoolScalarBCHW<24,24,12,12,1,6,2,2,2,2> total = 19575, with output window 15766
template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W>
void AvgpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W>::filter(
  input_window<TT>* in,      // BCHW (1x6x24x24)
  output_stream<TT>* restrict out     // BCPQ (1x6x12x12)
) {
  PROFILE_HEADER2;

  TT div_factor = inv(KH*KW);

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {

          TT sum = 0;
          for (int p = 0; p < KH; p++) {
            for (int q = 0; q < KW; q++) {
              TT a = window_readincr(in);
              sum += a;
            }
            window_incr(in, -KW+INP_W); // left KW, down 1
          }
          window_incr(in, -KH*INP_W + KW); // up KH, right KW
          writeincr(out, sum * div_factor);
        } // W
        window_incr(in, KH*INP_W - OUT_W*KW); // left OUT_W*KW, down KH
      } // H
    } // C
  } // B

  POOL_PROFILE_FOOTER("AvgpoolScalarBCHW");
}