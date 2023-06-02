#include "softmax.h"
#include "kernel_utils.h"


// https://nic.schraudolph.org/pubs/Schraudolph99.pdf 
double fastexp(float val) {
  int a = float2fix(1512775 * val, 0);
  long long b = (a + 1072632447);
  b <<= 32;
  double ans;
  memcpy(&ans, &b, sizeof(ans));
  return (float) ans;
}

// approximation with (1 + x/256)^256
float fastexp2(float val, int precision) {
  float ans = (1 + val*0.00390625);
  for (int i = 0; i < precision; i++)
    ans *= ans;
  return ans;
}

// taylor series for e^x  
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
    exp_scale = 1 / exp_scale;
    for (int j = 0; j < INP_W; j++) {
      window_writeincr(out, exp_v[j] * exp_scale);
    }
    window_incr(in, INP_W_PAD - INP_W);
    window_incr(out, INP_W_PAD - INP_W);
  }

  PROFILE_FOOTER;
}