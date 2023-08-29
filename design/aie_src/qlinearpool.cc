#include <limits>

#include "qlinearpool.h"
#include "kernel_utils.h"


#define QLINEARPOOL_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW);


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
void QLinearAvgpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW>::filter(
  input_window<TT>* in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  int resvi = 0;
  v16int16 resv = null_v16int16();

#define WRITE_OUT(res) \
  resv = upd_elem(resv, resvi, res); \
  if (resvi == 15) writeincr_v16(out, ((aie::vector<int16,16>) resv).pack<TT>()); \
  resvi = (resvi + 1) & 0xf;

  float scale = in_scale * inv(KH*KW * out_scale);

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {

          int sum = -in_zero *KH*KW;
          for (int p = 0; p < KH; p++) {
            for (int q = 0; q < KW; q++) {
              sum += window_readincr(in);
            }
            window_incr(in, -KW+INP_W); // left KW, down 1
          }
          window_incr(in, -KH*INP_W + KW); // up KH, right KW
          
          sum = round(sum * scale) + out_zero;
          if ((std::is_same<TT, int8_t>::value)) {
            sum = std::min(std::max(sum, -128), 127);
          } else {
            sum = std::min(std::max(sum, 0), 255);
          }

          WRITE_OUT(sum);
        } // W
        window_incr(in, -OUT_W*KW  + KH*INP_W); // down KH, left OUT_W*KW , account for padding
      } // H
    } // C
  } // B
#undef WRITE_OUT

  QLINEARPOOL_PROFILE_FOOTER("QLinearAvgpoolScalarBCHW");
}


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
QLinearGlobalAvgpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW>::QLinearGlobalAvgpoolScalarBCHW(
  float in_scale,
  float out_scale,
  int8_t in_zero,
  int8_t out_zero
): in_scale(in_scale), out_scale(out_scale), in_zero(in_zero), out_zero(out_zero) {
  float fscale = in_scale / (KH*KW*out_scale);
  BITSHIFT = 15 - log(fscale)/log(2); // int16_t scale
  scale = float2fix(fscale, BITSHIFT);
}


template <typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
void QLinearGlobalAvgpoolScalarBCHW<TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW>::filter(
  input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  using v128 = typename std::conditional<(std::is_same<TT, int8_t>::value), v128int8, v128uint8>::type;
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  v128 data = aie::zeros<TT,128>();
  v32 coef = aie::zeros<TT,32>();
  coef = upd_v(coef, 0, aie::broadcast<TT,16>(1));

  v16int16 res = null_v16int16();
  v16int16 select_mask = null_v16int16();

  for (int i = 0; i < KW % 16; i++) {
    select_mask = upd_elem(select_mask, i, 1);
  }
  aie::accum<acc48,16> shift_accum;
  shift_accum.from_vector(aie::broadcast<int32_t,16>(out_zero), BITSHIFT);

  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c+=16) {

      for (int i = 0; i < 16; i++) {

        int sum = -in_zero *KH*KW;
        v16acc48 acc = null_v16acc48();
        
        for (int p = 0; p <= KH-8; p+=8) {
          for (int j = 0; j < 8; j++)
            data = upd_v(data, j, readincr_v16(in));
          acc = mac16(acc, data, 0, 0x33323130, 32, 0x3120, coef, 0, 0x0, 0, 0x0);
        }
        data = aie::zeros<TT,128>();
        for (int j = 0; j < KH % 8; j++)
          data = upd_v(data, j, readincr_v16(in));
        acc = mac16(acc, data, 0, 0x33323130, 32, 0x3120, coef, 0, 0x0, 0, 0x0);
        
        acc = mul(srs(acc, 0), select_mask);
        sum += aie::reduce_add((aie::vector<int16_t,16>) srs(acc, 0)); 
        res = upd_elem(res, i, sum);
      }
      
      res = aie::mac(shift_accum, (aie::vector<int16_t,16>) res, scale).to_vector<int16_t>(BITSHIFT);
      writeincr_v16(out, ((aie::vector<int16,16>) res).pack<TT>());
    } // C
  } // B
#undef WRITE_OUT

  QLINEARPOOL_PROFILE_FOOTER("QLinearGlobalAvgpoolScalarBCHW");
}