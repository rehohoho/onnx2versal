#include <limits>
#include "qlinearsoftmax.h"
#include "kernel_utils.h"


#define QLINEARSOFTMAX_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s, %d,%d,%d>", \
    filter_name, typeid(TT).name(), INP_H, INP_W, INP_W_PAD);


float fastexp2(float val, int precision) {
  if (val < -500) return 0;
  float ans = (1 + val*0.00390625);
  for (int i = 0; i < precision; i++)
    ans *= ans;
  return ans;
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
void QLinearSoftmaxScalar<TT, INP_H, INP_W, INP_W_PAD>::filter(
	input_window<TT>* in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  float exp_v[INP_W];
  float exp_sum;
  float y_scale_inv = inv(y_scale);

  int resvi = 0;
  v16int16 resv = null_v16int16();

#define WRITE_OUT(res) \
  resv = upd_elem(resv, resvi, res); \
  if (resvi == 15) writeincr_v16(out, ((aie::vector<int16,16>) resv).pack<TT>()); \
  resvi = (resvi + 1) & 0xf;

  for (int i = 0; i < INP_H; i++) {
    // offset by max for exponentiation stability, assumes inputs in small range
    TT c = window_readincr(in);
    for (int j = 1; j < INP_W; j++) {
      c = max(c, window_readincr(in));
    }
    window_incr(in, -INP_W);

    exp_sum = 0;
    for (int j = 0; j < INP_W; j++) {
      float x = (window_readincr(in) - c) * x_scale; // x_zero is redundant due to divide
      exp_v[j] = fastexp2(x, 8);                     // x in [-inf, 0] from -c
      exp_sum += exp_v[j];
    }
    exp_sum = inv(exp_sum);
    for (int j = 0; j < INP_W; j++) {
      int y = round(exp_v[j] * exp_sum * y_scale_inv) + y_zero;
      if ((std::is_same<TT, int8_t>::value)) {
        y = std::min(std::max(y, -128), 127);
      } else {
        y = std::min(std::max(y, 0), 255);
      }
      WRITE_OUT(y);
    }
    window_incr(in, INP_W_PAD - INP_W);
    for (int j = 0; j < INP_W_PAD - INP_W; j++) {
      WRITE_OUT(0);
    }
  }
#undef WRITE_OUT

  QLINEARSOFTMAX_PROFILE_FOOTER("QLinearSoftmaxScalar");
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
QLinearSoftmaxFloatmul<TT, INP_H, INP_W, INP_W_PAD>::QLinearSoftmaxFloatmul (
  float x_scale,
  float y_scale,
  TT x_zero,
  TT y_zero
): x_scale(x_scale), y_scale(y_scale), x_zero(x_zero), y_zero(y_zero) {
  fastexp_scale = x_scale/256;
  min_value = max(ceil(-500 / x_scale), -32768);
};


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
void QLinearSoftmaxFloatmul<TT, INP_H, INP_W, INP_W_PAD>::filter(
	input_window<TT>* in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  float exp_v[INP_W_PAD];
  float *exp_v_ptr;
  int select_mask = (1 << (16 - INP_W_PAD + INP_W)) - 1;

  aie::vector<float,8> x1 = undef_v8float();
  aie::vector<int16_t,16> raw_in_v = undef_v16int16();

  aie::accum<accfloat,8> fastexp_shifts;
  fastexp_shifts.from_vector(aie::broadcast<float,8>(1), 0);
  aie::accum<acc48,16> y_zeros;
  y_zeros.from_vector(aie::broadcast<int16_t,16>(y_zero), EXP_BITSHIFT+OUT_BITSHIFT-12);

#define UPDATE_EXP(x1) \
  x1 = aie::mac(fastexp_shifts, x1, fastexp_scale).to_vector<float>(0); \
  for (int k = 0; k < 8; k++) \
    x1 = aie::mul_square(x1); \
  sum = aie::add(sum, x1); \
  aie::store_v(exp_v_ptr, x1); exp_v_ptr += 8;

  for (int i = 0; i < INP_H; i++) {
    aie::vector<float,8> sum = aie::zeros<float,8>();
    exp_v_ptr = (float *) exp_v;

    // offset by max for exponentiation stability, assumes inputs in small range
    aie::vector<TT,16> vmax = aie::broadcast<TT,16>(std::numeric_limits<TT>::lowest());
    for (int j = 0; j < INP_W_PAD - 16; j+=16) {
      vmax = aie::max(vmax, (aie::vector<TT,16>) window_readincr_v16(in));
    }
    int16_t max_offset = aie::reduce_max(vmax);
    for (int i = 0; i < INP_W % 16; i++) {
      max_offset = max(max_offset, window_readincr(in));
    }
    window_incr(in, -INP_W);

    for (int j = 0; j < INP_W_PAD - 16; j+=16) {
      raw_in_v = unpack(window_readincr_v16(in));
      raw_in_v = aie::sub(raw_in_v, max_offset);
      raw_in_v = aie::max(raw_in_v, min_value);

      x1 = fix2float(ext_v(raw_in_v, 0), 0);
      UPDATE_EXP(x1);
      x1 = fix2float(ext_v(raw_in_v, 1), 0);
      UPDATE_EXP(x1);
    }
    
    raw_in_v = unpack(window_readincr_v16(in));
    raw_in_v = aie::sub(raw_in_v, max_offset);
    raw_in_v = aie::max(raw_in_v, min_value);
    
    // ensure padding = 0
    v32int16 fat_res = null_v32int16();
    fat_res = upd_w(fat_res, 0, raw_in_v);
    fat_res = select32(select_mask, aie::zeros<int16_t, 32>(), fat_res);
    raw_in_v = ext_w(fat_res, 0);

    x1 = fix2float(ext_v(raw_in_v, 0), 0);
    UPDATE_EXP(x1);
    
    x1 = fix2float(ext_v(raw_in_v, 1), 0);
    UPDATE_EXP(x1);
    
    float expsum = INP_W - INP_W_PAD + aie::reduce_add(sum);
    expsum = inv(expsum * y_scale);
    int32_t intexpsum = float2fix(expsum, OUT_BITSHIFT); // necessary for precision

    exp_v_ptr = (float *) exp_v;
    for (int j = 0; j < INP_W; j+=16) {
      v16int32 res = null_v16int32();
      x1 = aie::load_v<8>(exp_v_ptr); exp_v_ptr += 8;
      res = upd_w(res, 0, float2fix(x1, EXP_BITSHIFT));
      x1 = aie::load_v<8>(exp_v_ptr); exp_v_ptr += 8;
      res = upd_w(res, 1, float2fix(x1, EXP_BITSHIFT));
      res = aie::mul((aie::vector<int32_t,16>) res, intexpsum).to_vector<int32_t>(12); // acc48 to int32
      auto resacc = aie::add(y_zeros, (aie::vector<int32_t,16>) res);
      writeincr_v16(out, resacc.to_vector<TT>(EXP_BITSHIFT+OUT_BITSHIFT-12));
    }
  
  } // INP_H
#undef UPDATE_EXP

  QLINEARSOFTMAX_PROFILE_FOOTER("QLinearSoftmaxFloatmul");
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
QLinearSoftmaxSingleaxis<TT, INP_H, INP_W, INP_W_PAD>::QLinearSoftmaxSingleaxis (
  float x_scale,
  float y_scale,
  TT x_zero,
  TT y_zero
): x_scale(x_scale), y_scale(y_scale), x_zero(x_zero), y_zero(y_zero) {
  
  EXP_BITSHIFT = 15 - log(x_scale/256) * inv(log(2)); // int16_t fastexp_scale
  EXP_BITSHIFT = min(EXP_BITSHIFT, 24);               // max 24 since acc48 used for squaring
  fastexp_scale = float2fix(x_scale/256, EXP_BITSHIFT);
  min_value = max(ceil(-500 / x_scale), -32768);
  expsum_offset = 1 << EXP_BITSHIFT;
};


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
void QLinearSoftmaxSingleaxis<TT, INP_H, INP_W, INP_W_PAD>::filter(
	input_window<TT>* in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  int32_t exp_v[INP_W_PAD];
  int32_t exp_sum;
  int32_t *exp_v_ptr;
  int select_mask = (1 << (16 - INP_W_PAD + INP_W)) - 1;

  aie::vector<int16_t,16> raw_in_v;
  aie::vector<int32_t,16> in_v;

  // quantization
  aie::accum<acc48,16> fastexp_shifts;
  aie::accum<acc48,16> y_zeros;
  fastexp_shifts.from_vector(aie::broadcast<int32_t,16>(1), EXP_BITSHIFT);
  y_zeros.from_vector(aie::broadcast<TT,16>(y_zero), EXP_BITSHIFT+OUT_BITSHIFT);

  for (int i = 0; i < INP_H; i++) {
    exp_v_ptr = (int32_t *) exp_v;
    aie::vector<int32_t,16> sum = aie::zeros<int32_t,16>();

    // offset by max for exponentiation stability, assumes inputs in small range
    aie::vector<TT,16> vmax = aie::broadcast<TT,16>(std::numeric_limits<TT>::lowest());
    for (int j = 0; j < INP_W/16*16; j+=16) {
      vmax = aie::max(vmax, (aie::vector<TT,16>) window_readincr_v16(in));
    }
    int16_t max_offset = aie::reduce_max(vmax);
    for (int i = 0; i < INP_W % 16; i++) {
      max_offset = max(max_offset, window_readincr(in));
    }
    window_incr(in, -INP_W);

    for (int j = 0; j < INP_W_PAD - 16; j+=16) {
      raw_in_v = unpack(window_readincr_v16(in));
      raw_in_v = aie::sub(raw_in_v, max_offset);
      raw_in_v = aie::max(raw_in_v, min_value);
      
      // in_v will be close to 1<<EXP_BITSHIFT usually, since exp(x) ~= (1 + x/256)**256,
      in_v = aie::mac(fastexp_shifts, raw_in_v, fastexp_scale).to_vector<int32_t>(0);
      for (int k = 0; k < 8; k++) chess_flatten_loop {
        in_v = aie::mul_square(in_v).to_vector<int32_t>(EXP_BITSHIFT);
      } // exp(x)
      
      sum = aie::add(sum, in_v);
      aie::store_v(exp_v_ptr, in_v); exp_v_ptr += 16;
    }
    
    raw_in_v = unpack(window_readincr_v16(in));
    raw_in_v = aie::sub(raw_in_v, max_offset);
    raw_in_v = aie::max(raw_in_v, min_value);
    
    // ensure padding = 0
    v32int16 fat_res = null_v32int16();
    fat_res = upd_w(fat_res, 0, raw_in_v);
    fat_res = select32(select_mask, aie::zeros<int16_t, 32>(), fat_res);
    raw_in_v = ext_w(fat_res, 0);
    
    // in_v will be close to 1<<EXP_BITSHIFT usually, since exp(x) ~= (1 + x/256)**256,
    in_v = aie::mac(fastexp_shifts, raw_in_v, fastexp_scale).to_vector<int32_t>(0);

    for (int k = 0; k < 8; k++) chess_flatten_loop {
      in_v = aie::mul_square(in_v).to_vector<int32_t>(EXP_BITSHIFT);
    } // exp(x)
    
    sum = aie::add(sum, in_v);
    aie::store_v(exp_v_ptr, in_v); exp_v_ptr += 16;

    exp_sum = (INP_W - INP_W_PAD) * expsum_offset;
    exp_sum += aie::reduce_add(sum); // outputs INP_W_PAD*(exp(x)>>EXP_BITSHIFT), may OOB int32_t
    float fexp_scale = fix2float(exp_sum, EXP_BITSHIFT);
    int16_t out_scale = float2fix(inv(fexp_scale * y_scale), OUT_BITSHIFT);

    exp_v_ptr = (int32_t *) exp_v;
    for (int j = 0; j < INP_W_PAD; j+=16) {
      in_v = aie::load_v<16>(exp_v_ptr); exp_v_ptr += 16;
      auto acc = aie::mac(y_zeros, in_v, out_scale);
      auto outvec = acc.to_vector<TT>(EXP_BITSHIFT + OUT_BITSHIFT);
      writeincr_v16(out, outvec);
    }
  } // INP_H

  QLINEARSOFTMAX_PROFILE_FOOTER("QLinearSoftmaxSingleaxis");
}
