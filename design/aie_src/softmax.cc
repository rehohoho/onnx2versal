#include "softmax.h"
#include "kernel_utils.h"


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
      exp_v[j] = expf(a);
      exp_scale += exp_v[j];
      printf("%f ", exp_v[j]);
    }
    printf("\n");
    exp_scale = 1 / exp_scale;
    for (int j = 0; j < INP_W; j++) {
      window_writeincr(out, exp_v[j] * exp_scale);
    }
    window_incr(in, INP_W_PAD - INP_W);
    window_incr(out, INP_W_PAD - INP_W);
  }

  PROFILE_FOOTER;
}