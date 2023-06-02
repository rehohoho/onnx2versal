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

  float exp_v[INP_W];
  float exp_scale;

  for (int i = 0; i < INP_H; i++) {
    exp_scale = 0;
    for (int j = 0; j < INP_W; j++) {
      float a = window_readincr(in);
      exp_v[j] = fastexp2(a, 8);
      exp_scale += exp_v[j];
    }
    exp_scale = inv(exp_scale);
    for (int j = 0; j < INP_W; j++) {
      window_writeincr(out, exp_v[j] * exp_scale);
    }
    window_incr(in, INP_W_PAD - INP_W);
    window_incr(out, INP_W_PAD - INP_W);
  }

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int INP_W_PAD>
void SoftmaxVector<INP_H, INP_W, INP_W_PAD>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running SoftmaxVector<%d,%d,%d>\n", INP_H, INP_W, INP_W_PAD));

  float scale = 0.00390625;
  float exp_v[INP_W_PAD];
  float exp_scale;

  float *exp_v_ptr;
  float *out_ptr = (float *) out->ptr;

  aie::vector<float,8> in_v;
  aie::accum<accfloat,8> ones;
  ones.from_vector(aie::broadcast<float,8>(1), 0);

  for (int i = 0; i < INP_H; i++) {
    exp_scale = INP_W - INP_W_PAD;
    exp_v_ptr = (float *) exp_v;
    
    for (int j = 0; j < INP_W; j+=8) {  
      in_v = window_readincr_v8(in);

      // compute using fastexp2 method
      in_v = aie::mac(ones, in_v, scale).to_vector<float>(0);
      for (int k = 0; k < 8; k++)
        in_v = aie::mul_square(in_v);
      aie::store_v(exp_v_ptr, in_v); exp_v_ptr += 8;
      exp_scale += aie::reduce_add(in_v);
    }

    exp_scale = inv(exp_scale);
    
    exp_v_ptr = (float *) exp_v;
    for (int j = 0; j < INP_W; j+=8) {
      in_v = aie::load_v<8>(exp_v_ptr); exp_v_ptr += 8;
      in_v = aie::mul(in_v, exp_scale);
      aie::store_v(out_ptr, in_v); out_ptr += 8;
    }
  }

  PROFILE_FOOTER;
}
