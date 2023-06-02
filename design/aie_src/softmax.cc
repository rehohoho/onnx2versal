#include "softmax.h"
#include "kernel_utils.h"


double fastexp(float val) {
  int a = float2fix(1512775 * val, 0);
  long long b = (a + 1072632447);
  b <<= 32;
  double ans;
  memcpy(&ans, &b, sizeof(ans));
  return (float) ans;
}


float fastexp2(float val, int precision) {
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
  PROFILE_HEADER(printf(
    "Running SoftmaxScalar<%d,%d,%d>\n", INP_H, INP_W, INP_W_PAD));

  float exp_v1[INP_W];
  float exp_scale1;

  for (int i = 0; i < INP_H; i++) {
    exp_scale1 = 0;
    for (int j = 0; j < INP_W; j++) {
      float a = window_readincr(in);
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

  PROFILE_FOOTER;
}


// 1417 with single aie::mac_square
// 1082 with two fpmacs
template <int INP_H, int INP_W, int INP_W_PAD>
void SoftmaxVector<INP_H, INP_W, INP_W_PAD>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running SoftmaxVector<%d,%d,%d>\n", INP_H, INP_W, INP_W_PAD));

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

  PROFILE_FOOTER;
}
