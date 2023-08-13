#include "softmax.h"
#include "kernel_utils.h"


#define SOFTMAX_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d>", \
    filter_name, INP_H, INP_W, INP_W_PAD);

double fastexp(float val) {
  int a = float2fix(1512775 * val, 0);
  long long b = (a + 1072632447);
  b <<= 32;
  double ans;
  memcpy(&ans, &b, sizeof(ans));
  return (float) ans;
}


float fastexp2(float val, int precision) {
  if (val < -500) return 0;
  float ans = (1 + val*0.00390625);
  for (int i = 0; i < precision; i++)
    ans *= ans;
  return ans;
}

template <int INP_H, int INP_W, int INP_W_PAD>
float SoftmaxScalar<INP_H, INP_W, INP_W_PAD>::fastexp3(float val, int precision) {
  float x = val;
  float ans = 1;
  for (int i = 0; i < precision; i++) {
    ans += x * coef[i];
    x *= x;
  }
  return ans;
}


template <int INP_H, int INP_W, int INP_W_PAD>
void SoftmaxScalar<INP_H, INP_W, INP_W_PAD>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER2;

  float exp_v1[INP_W];
  float exp_scale1;

  for (int i = 0; i < INP_H; i++) {
    exp_scale1 = 0;
    
    // offset by max for exponentiation stability, assumes inputs in small range
    float c = window_readincr(in);
    for (int j = 1; j < INP_W; j++) {
      c = max(c, window_readincr(in));
    }
    window_incr(in, -INP_W);

    for (int j = 0; j < INP_W; j++) {
      float a = window_readincr(in) - c;
      exp_v1[j] = fastexp2(a, 8);
      exp_scale1 += exp_v1[j];
    }
    exp_scale1 = inv(exp_scale1);
    for (int j = 0; j < INP_W; j++) {
      window_writeincr(out, exp_v1[j] * exp_scale1);
    }
    window_incr(in, INP_W_PAD - INP_W);
    window_incr(out, INP_W_PAD - INP_W);
  }

  SOFTMAX_PROFILE_FOOTER("SoftmaxScalar");
}


template <int INP_H, int INP_W, int INP_W_PAD>
void SoftmaxSingleaxis<INP_H, INP_W, INP_W_PAD>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER2;

  float *out_ptr = (float *) out->ptr; 

  float exp_v[INP_W_PAD];
  float scale = 0.00390625;
  float exp_scale;
  float *exp_v_ptr;

  int select_mask = (INP_W_PAD % 8 == 0) ? (1 << (8 - INP_W_PAD + INP_W)) - 1 : (1 << (4 - INP_W_PAD + INP_W)) - 1;
  v16float res = null_v16float();

  aie::vector<float,8> in_v;
  aie::accum<accfloat,8> ones;
  ones.from_vector(aie::broadcast<float,8>(1), 0);

  for (int i = 0; i < INP_H; i++) {
    exp_v_ptr = (float *) exp_v;

    // offset by max for exponentiation stability, assumes inputs in small range
    v8float vmax = window_readincr_v8(in);
    for (int j = 8; j < INP_W/8*8; j+=8) {
      vmax = fpmax(vmax, window_readincr_v8(in));
    }
    float max_offset = aie::reduce_max((aie::vector<float,8>) vmax);
    for (int i = 0; i < INP_W % 8; i++) {
      max_offset = max(max_offset, window_readincr(in));
    }
    window_incr(in, -INP_W);

    aie::vector<float,8> sum = aie::zeros<float,8>();

    for (int j = 0; j < INP_W_PAD - 8; j+=8) {  // INP_W_PAD % 8
      in_v = window_readincr_v8(in);
      in_v = aie::sub(in_v, max_offset);
      in_v = aie::max(in_v, -500.f);

      // compute using fastexp2 method
      in_v = aie::mac(ones, in_v, scale).to_vector<float>(0);
      for (int k = 0; k < 8; k++)
        in_v = aie::mul_square(in_v);
      
      sum = aie::add(sum, in_v);
      aie::store_v(exp_v_ptr, in_v); exp_v_ptr += 8;
    }

    in_v = window_readincr_v8(in);
    in_v = aie::sub(in_v, max_offset);
    in_v = aie::max(in_v, -500.f);

    // ensure padding = 0
    res = upd_w(res, 1, in_v);
    res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
    in_v = ext_w(res, 0);

    in_v = aie::mac(ones, in_v, scale).to_vector<float>(0);
    for (int k = 0; k < 8; k++)
      in_v = aie::mul_square(in_v);
    
    sum = aie::add(sum, in_v);
    aie::store_v(exp_v_ptr, in_v); exp_v_ptr += 8;

    exp_scale = INP_W - INP_W_PAD; // offset for calculating by INP_W_PAD
    exp_scale += aie::reduce_add(sum);
    exp_scale = inv(exp_scale);

    exp_v_ptr = (float *) exp_v;
    for (int j = 0; j < INP_W_PAD; j+=8) {  // INP_W_PAD % 8
      in_v = aie::load_v<8>(exp_v_ptr); exp_v_ptr += 8;
      in_v = aie::mul(in_v, exp_scale);
      aie::store_v(out_ptr, in_v); out_ptr += 8;
    }
  }

  SOFTMAX_PROFILE_FOOTER("SoftmaxSingleaxis");
}


template <int INP_H, int INP_W, int INP_W_PAD>
void SoftmaxMultiaxis<INP_H, INP_W, INP_W_PAD>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER2;

  float exp_v1[INP_W_PAD];
  float exp_v2[INP_W_PAD];
  float exp_scale1;
  float exp_scale2;

  v8float *exp_v1_ptr;
  v8float *exp_v2_ptr;
  v8float *out_ptr = (v8float *) out->ptr;

  v16float data = undef_v16float();
  v8float x1 = undef_v8float();
  v8float x2 = undef_v8float();
  v8float ones = aie::broadcast<float,8>(1);
  v8float scale = aie::broadcast<float,8>(0.00390625);

  for (int i = 0; i < INP_H; i+=2) {
    exp_scale1 = INP_W - INP_W_PAD;
    exp_scale2 = INP_W - INP_W_PAD;
    exp_v1_ptr = (v8float *) exp_v1;
    exp_v2_ptr = (v8float *) exp_v2;

    v8float exp_v1_sum = null_v8float();
    v8float exp_v2_sum = null_v8float();
    
    for (int j = 0; j < INP_W_PAD; j+=8) {  
      x1 = window_read_v8(in);
      x1 = fpmac(ones, x1, scale);
      window_incr(in, INP_W_PAD);
      x2 = window_read_v8(in);
      x2 = fpmac(ones, x2, scale);
      window_incr(in, -INP_W_PAD+8);

      // compute using fastexp2 method
      for (int k = 0; k < 8; k++) chess_flatten_loop {
        data = upd_w(data, 0, x1);
        x1 = fpmul(data, 0, 0x76543210, 0, 0x76543210);
        data = upd_w(data, 1, x2);
        x2 = fpmul(data, 8, 0x76543210, 8, 0x76543210);
      }
      exp_v1_sum = fpadd(exp_v1_sum, x1);
      *exp_v1_ptr = x1; exp_v1_ptr++;
      exp_v2_sum = fpadd(exp_v2_sum, x2);
      *exp_v2_ptr = x2; exp_v2_ptr++;
    }

    exp_scale1 += aie::reduce_add((aie::vector<float,8>) exp_v1_sum);
    exp_scale1 = inv(exp_scale1);
    exp_v1_ptr = (v8float *) exp_v1;
    exp_scale2 += aie::reduce_add((aie::vector<float,8>) exp_v2_sum);
    exp_scale2 = inv(exp_scale2);
    exp_v2_ptr = (v8float *) exp_v2;

    v8float exp_scale1_v = aie::broadcast<float, 8>(exp_scale1);
    v8float exp_scale2_v = aie::broadcast<float, 8>(exp_scale2);

    for (int j = 0; j < INP_W_PAD; j+=8) {
      x1 = fpmul(*exp_v1_ptr, exp_scale1_v); exp_v1_ptr++;
      *out_ptr = x1; out_ptr += INP_W_PAD/8;
      x2 = fpmul(*exp_v2_ptr, exp_scale2_v); exp_v2_ptr++;
      *out_ptr = x2; out_ptr += -INP_W_PAD/8 + 1;
    }
    
    out_ptr += INP_W_PAD/8;
    window_incr(in, INP_W_PAD);
  }

  SOFTMAX_PROFILE_FOOTER("SoftmaxMultiaxis");
}
