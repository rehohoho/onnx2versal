#include "qlinearsoftmax.h"
#include "kernel_utils.h"


#define QLINEARSOFTMAX_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s, %d,%d,%d>", \
    filter_name, typeid(TT).name(), INP_H, INP_W, INP_W_PAD);


float fastexp2(float val, int precision) {
  float ans = (1 + val*0.00390625);
  for (int i = 0; i < precision; i++)
    ans *= ans;
  return ans;
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
void QLinearSoftmaxScalar<TT, INP_H, INP_W, INP_W_PAD>::filter(
	input_window<TT>* in,
  output_stream<TT>* out
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
    exp_sum = 0;
    for (int j = 0; j < INP_W; j++) {
      float x = window_readincr(in);
      x = (x - x_zero) * x_scale;
      exp_v[j] = fastexp2(x, 8); // can over and underflow
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
  fastexp_shift = -x_zero*x_scale/256+1;
};


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
void QLinearSoftmaxFloatmul<TT, INP_H, INP_W, INP_W_PAD>::filter(
	input_window<TT>* in,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  float exp_v[INP_W_PAD];
  float c = 0.00390625;
  float y_scale_inv = inv(y_scale);
  float expsum;
  
  float *exp_v_ptr;

  aie::vector<float,8> x1 = undef_v8float();
  aie::vector<float,8> x2 = undef_v8float();
  aie::vector<int32_t,8> intx1 = undef_v8int32();
  aie::vector<int32_t,8> intx2 = undef_v8int32();
  v16int16 raw_in_v = undef_v16int16();

  aie::accum<accfloat,8> fastexp_shifts;
  fastexp_shifts.from_vector(aie::broadcast<float,8>(fastexp_shift), 0);
  
  aie::accum<acc48,8> y_zeros;
  aie::accum<acc48,8> res1;
  aie::accum<acc48,8> res2;
  y_zeros.from_vector(aie::broadcast<int16_t,8>(y_zero), EXP_BITSHIFT+OUT_BITSHIFT-12);

  for (int i = 0; i < INP_H; i++) {
    aie::vector<float,8> sum = aie::zeros<float,8>();
    exp_v_ptr = (float *) exp_v;

    for (int j = 0; j < INP_W_PAD; j+=16) {

      raw_in_v = unpack(window_readincr_v16(in));
      x1 = fix2float(ext_v(raw_in_v, 0), 0);
      x1 = aie::mac(fastexp_shifts, x1, fastexp_scale).to_vector<float>(0);
      
      x2 = fix2float(ext_v(raw_in_v, 1), 0);
      x2 = aie::mac(fastexp_shifts, x2, fastexp_scale).to_vector<float>(0);

      for (int k = 0; k < 8; k++) {
        x1 = aie::mul_square(x1);
        x2 = aie::mul_square(x2);
      }
      sum = aie::add(sum, x1);
      aie::store_v(exp_v_ptr, x1); exp_v_ptr += 8;
      sum = aie::add(sum, x2);
      aie::store_v(exp_v_ptr, x2); exp_v_ptr += 8;
    }
    
    expsum = INP_W - INP_W_PAD + aie::reduce_add(sum);
    expsum = inv(expsum) * y_scale_inv;
    int32_t intexpsum = float2fix(expsum, OUT_BITSHIFT); // necessary for precision

    exp_v_ptr = (float *) exp_v;
    for (int j = 0; j < INP_W; j+=16) {
      x1 = aie::load_v<8>(exp_v_ptr); exp_v_ptr += 8;
      intx1 = float2fix(x1, EXP_BITSHIFT);
      intx1 = aie::mul(intx1, intexpsum).to_vector<int32_t>(12); // acc48 to int32
      res1 = aie::add(y_zeros, intx1);
      
      x2 = aie::load_v<8>(exp_v_ptr); exp_v_ptr += 8;
      intx2 = float2fix(x2, EXP_BITSHIFT);
      intx2 = aie::mul(intx2, intexpsum).to_vector<int32_t>(12);
      res2 = aie::add(y_zeros, intx2);

      auto res = aie::concat(res1, res2);
      writeincr_v16(out, res.to_vector<TT>(EXP_BITSHIFT+OUT_BITSHIFT-12));
    }
  
  } // INP_H

  QLINEARSOFTMAX_PROFILE_FOOTER("QLinearSoftmaxFloatmul");
}


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
QLinearSoftmaxSingleaxis<TT, INP_H, INP_W, INP_W_PAD>::QLinearSoftmaxSingleaxis (
  float x_scale,
  float y_scale,
  TT x_zero,
  TT y_zero
): x_scale(x_scale), y_scale(y_scale), x_zero(x_zero), y_zero(y_zero) {
  
  fastexp_scale = float2fix(x_scale/256, EXP_BITSHIFT);
  fastexp_shift = float2fix(-x_zero*x_scale/256+1, EXP_BITSHIFT);

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero
  
  aie::vector<int32_t,16> in_v = aie::broadcast<int32_t,16>(x_zero);
  aie::accum<acc48,16> fastexp_shifts;
  fastexp_shifts.from_vector(aie::broadcast<int32_t,16>(fastexp_shift), 0);
  
  in_v = aie::mac(fastexp_shifts, in_v, fastexp_scale).to_vector<int32_t>(0);
  for (int k = 0; k < 8; k++) chess_flatten_loop {
    in_v = aie::mul_square(in_v).to_vector<int32_t>(EXP_BITSHIFT);
  } // exp(x)
  expsum_offset = (int32_t) ext_elem(in_v, 0); // for in[i] = 0
};


template <typename TT, int INP_H, int INP_W, int INP_W_PAD>
void QLinearSoftmaxSingleaxis<TT, INP_H, INP_W, INP_W_PAD>::filter(
	input_window<TT>* in,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  int32_t exp_v[INP_W_PAD];
  int32_t exp_sum;
  int16_t out_scale;
  int32_t *exp_v_ptr;

  aie::vector<int16_t,16> raw_in_v;
  aie::vector<int32_t,16> in_v;

  // quantization
  aie::accum<acc48,16> fastexp_shifts;
  aie::accum<acc48,16> y_zeros;
  fastexp_shifts.from_vector(aie::broadcast<int32_t,16>(fastexp_shift), 0);
  y_zeros.from_vector(aie::broadcast<TT,16>(y_zero), EXP_BITSHIFT+OUT_BITSHIFT);

  for (int i = 0; i < INP_H; i++) {
    exp_v_ptr = (int32_t *) exp_v;
    aie::vector<int32_t,16> sum = aie::zeros<int32_t,16>();

    for (int j = 0; j < INP_W_PAD; j+=16) {
      raw_in_v = unpack(window_readincr_v16(in));
      
      // in_v will be close to 1<<EXP_BITSHIFT usually, since exp(x) ~= (1 + x/256)**256,
      in_v = aie::mac(fastexp_shifts, raw_in_v, fastexp_scale).to_vector<int32_t>(0);
      for (int k = 0; k < 8; k++) chess_flatten_loop {
        in_v = aie::mul_square(in_v).to_vector<int32_t>(EXP_BITSHIFT);
      } // exp(x)
      
      sum = aie::add(sum, in_v);
      aie::store_v(exp_v_ptr, in_v); exp_v_ptr += 16;
    }

    exp_sum = (INP_W - INP_W_PAD) * expsum_offset;
    exp_sum += aie::reduce_add(sum); // outputs INP_W_PAD*(exp(x)>>EXP_BITSHIFT), may OOB int32_t
    float fexp_scale = fix2float(exp_sum, EXP_BITSHIFT);
    out_scale = float2fix(inv(fexp_scale * y_scale), OUT_BITSHIFT);

    exp_v_ptr = (int32_t *) exp_v;
    for (int j = 0; j < INP_W_PAD; j+=16) {
      in_v = aie::load_v<16>(exp_v_ptr); exp_v_ptr += 16;
      print_vec<int, int>((int *) &in_v, 16);
      auto acc = aie::mac(y_zeros, in_v, out_scale);
      auto outvec = acc.to_vector<TT>(EXP_BITSHIFT + OUT_BITSHIFT);
      writeincr_v16(out, outvec);
    }
  } // INP_H

  QLINEARSOFTMAX_PROFILE_FOOTER("QLinearSoftmaxSingleaxis");
}
