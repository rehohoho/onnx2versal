#include "quantize_linear.h"
#include "kernel_utils.h"


template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinearScalar<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearScalar<%d,%d,%d>\n", INP_H, INP_W, OUT_W));

  float y_scale_inv = 1/y_scale;

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j++) {
      float x = window_readincr(in);
      int y = round(x * y_scale_inv) + y_zero_point;
      y = std::min(std::max(y, -128), 128);
      window_writeincr(out, y);
    }
    window_incr(out, OUT_W-INP_W);
  }

  PROFILE_FOOTER;
}

//try use two?
template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinearVector<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearVector<%d,%d,%d>\n", INP_H, INP_W, OUT_W));
  
  float y_scale_inv = 1/y_scale;
  v8float y_scale_invvec = aie::broadcast<float, 8>(y_scale_inv);
  v8int32 y_zero = aie::broadcast<int32_t, 8>(y_zero_point);
  v8int32 y1 = null_v8int32();
  v8int32 y2 = null_v8int32();
  v8acc48 acc = null_v8acc48();
  v16int16 y = null_v16int16();

  for (int i = 0; i < INP_H; i++) {
    int j = 0;
    for (j; j < INP_W-16; j+=16) {
      v8float x1 = window_readincr_v8(in);
      x1 = fpmul(x1, y_scale_invvec);
      v8float x2 = window_readincr_v8(in);
      x2 = fpmul(x2, y_scale_invvec);

      y1 = float2fix(x1, 0);
      y1 = y1 + y_zero;
      y2 = float2fix(x2, 0);
      y2 = y2 + y_zero;

      acc = ups(y1, 0);
      y = upd_v(y, 0, srs(acc, 0));
      acc = ups(y2, 0);
      y = upd_v(y, 1, srs(acc, 0));

      window_writeincr(out, pack(y));
    }
    for (j; j < INP_W; j++) {
      float x = window_readincr(in);
      int y = round(x * y_scale_inv) + y_zero_point;
      y = std::min(std::max(y, -128), 128);
      window_writeincr(out, y);
    }
    window_incr(out, OUT_W-INP_W);
  }

  PROFILE_FOOTER;
}
