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

  float y_scale_inv = inv(y_scale);

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j++) {
      float x = window_readincr(in);
      int y = round(x * y_scale_inv) + y_zero;
      y = std::min(std::max(y, -128), 128);
      window_writeincr(out, y);
    }
    window_incr(out, OUT_W-INP_W);
  }

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int OUT_W>
QuantizeLinear<INP_H, INP_W, OUT_W>::QuantizeLinear(
  float y_scale,
  int8_t y_zero
): y_scale(y_scale), y_zero(y_zero) {
  ybitshift = 15 - log(inv(y_scale)) * inv(log(2)); // int16_t y_scale_inv_int
  assert(ybitshift < 32); // float2fix shift in [-32:31]
  printf("xshift %d, yshift %d\n", xbitshift, ybitshift);
  y_scale_inv_int = float2fix(inv(y_scale), ybitshift);
};


template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinear<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinear<%d,%d,%d>\n", INP_H, INP_W, OUT_W));
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  v16int32 y = null_v16int32();
  v16acc48 acc1 = null_v16acc48(); // two instances cause vector reg spilling
  aie::accum<acc48,16> acc_shift;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), xbitshift+ybitshift);

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j+=16) {
      v8float x1 = window_readincr_v8(in);
      y = upd_w(y, 0, float2fix(x1, xbitshift));
      v8float x2 = window_readincr_v8(in);
      y = upd_w(y, 1, float2fix(x2, xbitshift));
      acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) y, y_scale_inv_int);
      window_writeincr(out, bsrs(acc1, xbitshift+ybitshift));
    }
    window_incr(in, -(INP_W+15)/16*16 + INP_W);
  }

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinearFmul<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearFmul<%d,%d,%d>\n", INP_H, INP_W, OUT_W));
  
  int8_t *out_ptr = out->ptr;

  float y_scale_inv = inv(y_scale);
  v8float y_scale_invvec = aie::broadcast<float, 8>(y_scale_inv);
  v8int16 y_zerovec = aie::broadcast<int16_t, 8>(y_zero);
  
  v8acc48 acc1 = null_v8acc48();
  v8acc48 acc2 = null_v8acc48();
  v16acc48 acc = null_v16acc48();

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j+=16) {
      v8float x1 = window_readincr_v8(in);
      x1 = fpmul(x1, y_scale_invvec);
      v8float x2 = window_readincr_v8(in);
      x2 = fpmul(x2, y_scale_invvec);

      acc1 = ups(float2fix(x1, 0), 0);
      acc1 = acc1 + y_zerovec;
      acc = upd_lo(acc, acc1);

      acc2 = ups(float2fix(x2, 0), 0);
      acc2 = acc2 + y_zerovec;
      acc = upd_hi(acc, acc2);

      window_writeincr(out, bsrs(acc, 0));
    }
    window_incr(in, -(INP_W+15)/16*16 + INP_W);
  }

  PROFILE_FOOTER;
}