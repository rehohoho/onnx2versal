#include "quantize_linear.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinearScalar<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearScalar<%d,%d,%d>\n", INP_H, INP_W, OUT_W));

  float y_scale_inv = 1/y_scale;

  for (int i = 0; i < INP_H; i++) {
    int j = 0;
    for (j; j < INP_W; j++) {
      float x = window_readincr(in);
      int y = round(x * y_scale_inv) + y_zero_point;
      y = std::min(std::max(y, -128), 128);
      window_writeincr(out, y);
    }
    for (j; j < OUT_W; j++) 
      window_writeincr(out, y_zero_point);
  }

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int OUT_W>
QuantizeLinearVector<INP_H, INP_W, OUT_W>::QuantizeLinearVector(
  float y_scale,
  int y_zero_point
): y_scale(y_scale), y_zero_point(y_zero_point) {
  int width_r = INP_H % 16;
  select_mask = (1 << width_r) - 1;
  y_scale_inv_int = float2fix(1/y_scale, 0);
  shift = float2fix((float) y_zero_point, bitshift);
};


template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinearVector<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearVector<%d,%d,%d>\n", INP_H, INP_W, OUT_W));
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  v16int32 y = null_v16int32();
  aie::accum<acc48,16> aieacc1;

  for (int i = 0; i < INP_H; i++) {
    int j = 0;
    for (j; j < INP_W-16; j+=16) {
      v8float x1 = window_readincr_v8(in);
      y = upd_w(y, 0, float2fix(x1, bitshift));
      v8float x2 = window_readincr_v8(in);
      y = upd_w(y, 1, float2fix(x2, bitshift));
      aieacc1 = aie::mul<acc48>((aie::vector<int32_t,16>) y, y_scale_inv_int);
      aieacc1 = aie::add(aieacc1, shift);
      window_writeincr(out, bsrs((v16acc48) aieacc1, bitshift));
    }

    v8float x1 = window_readincr_v8(in);
    y = upd_w(y, 0, float2fix(x1, bitshift));
    v8float x2 = window_readincr_v8(in);
    y = upd_w(y, 1, float2fix(x2, bitshift));
    y = select16(select_mask, null_v16int32(), y);
    aieacc1 = aie::mul<acc48>((aie::vector<int32_t,16>) y, y_scale_inv_int);
    aieacc1 = aie::add(aieacc1, shift);
    
    window_writeincr(out, bsrs((v16acc48) aieacc1, bitshift));
    window_incr(in, -(INP_H+15)/16*16 + INP_W);
  }

  PROFILE_FOOTER;
}
