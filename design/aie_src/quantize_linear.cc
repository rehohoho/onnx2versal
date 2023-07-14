#include "quantize_linear.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define QUANTIZE_LINEAR_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d>", \
    filter_name, typeid(TT).name(), INP_H, INP_W, OUT_W);


template <typename TT, int INP_H, int INP_W, int OUT_W>
void QuantizeLinearScalar<TT, INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;

  float y_scale_inv = inv(y_scale);

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j++) {
      float x = window_readincr(in);
      int y = round(x * y_scale_inv) + y_zero;
      if ((std::is_same<TT, int8_t>::value)) {
        y = std::min(std::max(y, -128), 127);
      } else {
        y = std::min(std::max(y, 0), 255);
      }
      
      window_writeincr(out, y);
    }
    window_incr(out, OUT_W-INP_W);
  }

  QUANTIZE_LINEAR_PROFILE_FOOTER("QuantizeLinearScalar");
}


template <typename TT, int INP_H, int INP_W, int OUT_W>
QuantizeLinear<TT, INP_H, INP_W, OUT_W>::QuantizeLinear(
  float y_scale,
  int8_t y_zero
): y_scale(y_scale), y_zero(y_zero) {
  ybitshift = 15 - log(1/y_scale) / log(2); // int16_t y_scale_inv_int
  assert(ybitshift < 32); // float2fix shift in [-32:31]
  y_scale_inv_int = float2fix(inv(y_scale), ybitshift);
};


template <typename TT, int INP_H, int INP_W, int OUT_W>
void QuantizeLinear<TT, INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER2;
  
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

  QUANTIZE_LINEAR_PROFILE_FOOTER("QuantizeLinear");
}


template <typename TT, int INP_H, int INP_W, int OUT_W>
void QuantizeLinearFmul<TT, INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;
  
  v8float scale = aie::broadcast<float, 8>(inv(y_scale));
  v8int16 shift = aie::broadcast<int16_t, 8>(y_zero);
  
  v8acc48 acc1 = null_v8acc48();
  v8acc48 acc2 = null_v8acc48();
  v16acc48 acc = null_v16acc48();

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j+=16) {
      v8float x1 = window_readincr_v8(in);
      x1 = fpmul(x1, scale);
      v8float x2 = window_readincr_v8(in);
      x2 = fpmul(x2, scale);

      acc1 = ups(float2fix(x1, 0), 0);
      acc1 = acc1 + shift;
      acc = upd_lo(acc, acc1);

      acc2 = ups(float2fix(x2, 0), 0);
      acc2 = acc2 + shift;
      acc = upd_hi(acc, acc2);

      window_writeincr(out, ((aie::accum<acc48,16>) acc).to_vector<TT>(0));
    }
    window_incr(in, -(INP_W+15)/16*16 + INP_W);
  }

  QUANTIZE_LINEAR_PROFILE_FOOTER("QuantizeLinearFmul");
}


template <typename TT, int INP_H, int INP_W, int OUT_W>
void QuantizeLinearFmulStream<TT, INP_H, INP_W, OUT_W>::filter(
	input_stream<float>* in,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;
  
  v8float scale = aie::broadcast<float, 8>(inv(y_scale));
  v8int16 shift = aie::broadcast<int16_t, 8>(y_zero);

  v8float x1 = null_v8float();
  v8float x2 = null_v8float();
  
  v8acc48 acc1 = null_v8acc48();
  v8acc48 acc2 = null_v8acc48();
  v16acc48 acc = null_v16acc48();

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j <= INP_W-16; j+=16) {
      x1 = upd_v(x1, 0, readincr_v4(in));
      x1 = upd_v(x1, 1, readincr_v4(in));
      x1 = fpmul(x1, scale);
      x2 = upd_v(x2, 0, readincr_v4(in));
      x2 = upd_v(x2, 1, readincr_v4(in));
      x2 = fpmul(x2, scale);

      acc1 = ups(float2fix(x1, 0), 0);
      acc1 = acc1 + shift;
      acc = upd_lo(acc, acc1);

      acc2 = ups(float2fix(x2, 0), 0);
      acc2 = acc2 + shift;
      acc = upd_hi(acc, acc2);

      writeincr_v16(out, ((aie::accum<acc48,16>) acc).to_vector<TT>(0));
    }

    // handle for INP_W%16 = 4,8,12 since INP_W%4
    if (INP_W % 16 >= 4) x1 = upd_v(x1, 0, readincr_v4(in));
    if (INP_W % 16 >= 8) x1 = upd_v(x1, 1, readincr_v4(in));
    if (INP_W % 16 >= 4) {
      x1 = fpmul(x1, scale);
      acc1 = ups(float2fix(x1, 0), 0);
      acc1 = acc1 + shift;
      acc = upd_lo(acc, acc1);
    }
    if (INP_W % 16 >= 12) {
      x2 = upd_v(x2, 0, readincr_v4(in));
      x2 = fpmul(x2, scale);
      acc2 = ups(float2fix(x2, 0), 0);
      acc2 = acc2 + shift;
      acc = upd_hi(acc, acc2);
    }
    if (INP_W % 16 >= 4) {
      writeincr_v16(out, ((aie::accum<acc48,16>) acc).to_vector<TT>(0));
    }
  }

  QUANTIZE_LINEAR_PROFILE_FOOTER("QuantizeLinearFmulStream");
}