#include "dequantize_linear.h"
#include "kernel_utils.h"


#define DEQUANTIZE_LINEAR_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, INP_W, OUT_W);


template <typename TT, int B, int INP_W, int OUT_W>
void DequantizeLinearScalar<TT, B, INP_W, OUT_W>::filter(
	input_window<TT>* in,
  output_window<float>* out
) {
  PROFILE_HEADER2;

  for (int i = 0; i < B; i++) {
    for (int j = 0; j < OUT_W; j++) {
      int x = window_readincr(in);
      float y = (x - zero) * scale;
      window_writeincr(out, y);
    }
    window_incr(in, OUT_W - INP_W);
  }

  DEQUANTIZE_LINEAR_PROFILE_FOOTER("DequantizeLinearScalar");
}


template <typename TT, int B, int INP_W, int OUT_W>
DequantizeLinear<TT, B, INP_W, OUT_W>::DequantizeLinear (
  float scale,
  TT zero
): scale(scale), zero(zero) {
  bitshift = 15 - log(scale) * inv(log(2)); // int16_t y_scale_inv_int
  iscale = float2fix(scale, bitshift);
  ishift = float2fix(-zero * scale, bitshift);
}

template <typename TT, int B, int INP_W, int OUT_W>
void DequantizeLinear<TT, B, INP_W, OUT_W>::filter(
	input_window<TT>* in,
  output_window<float>* out
) {
  PROFILE_HEADER2;
  
  aie::accum<acc48, 16> acc_shift;
  acc_shift.from_vector(aie::broadcast<int32_t, 16>(ishift), 0);

  for (int i = 0; i < B; i++) {
    for (int j = 0; j <= OUT_W - 16; j+=16) {
      aie::vector<int16_t, 16> x = unpack(window_readincr_v16(in));
      v16int32 y = aie::mac(acc_shift, x, iscale).to_vector<int32_t>(0);
      
      v8int32 y1 = ext_w(y, 0);
      v8float y1f = fix2float(y1, bitshift);
      window_writeincr(out, y1f);

      v8int32 y2 = ext_w(y, 1);
      v8float y2f = fix2float(y2, bitshift);
      window_writeincr(out, y2f);
    }
    
    if (OUT_W % 16 != 0) {
      aie::vector<int16_t, 16> x = unpack(window_readincr_v16(in));
      v16int32 y = aie::mac(acc_shift, x, iscale).to_vector<int32_t>(0);
      v8int32 y1 = ext_w(y, 0);
      v8float y1f = fix2float(y1, bitshift);

      v8int32 y2 = ext_w(y, 1);
      v8float y2f = fix2float(y2, bitshift);
      if (OUT_W % 16 == 4)
        window_writeincr(out, ext_v(y1f, 0));
      if (OUT_W % 16 >= 8)
        window_writeincr(out, y1f);
      if (OUT_W % 16 == 12)
        window_writeincr(out, ext_v(y2f, 0));
    }
    
  }

  DEQUANTIZE_LINEAR_PROFILE_FOOTER("DequantizeLinear");
}
