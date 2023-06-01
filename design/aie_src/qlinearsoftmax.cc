#include "qlinearsoftmax.h"
#include "kernel_utils.h"


template <int INP_H, int INP_W, int INP_W_PAD>
void QlinearsoftmaxScalar<INP_H, INP_W, INP_W_PAD>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QlinearsoftmaxScalar<%d,%d,%d>\n", INP_H, INP_W, INP_W_PAD));

  float exp_v[INP_W];
  float exp_scale;
  float y_scale_inv = 1 / y_scale;

  for (int i = 0; i < INP_H; i++) {
    exp_scale = 0;
    for (int j = 0; j < INP_W; j++) {
      float x = window_readincr(in);
      x = (x - x_zero) * x_scale;
      exp_v[j] = expf(x); // can over and underflow
      exp_scale += exp_v[j];
    }
    exp_scale = 1 / exp_scale;
    for (int j = 0; j < INP_W; j++) {
      int y = round(exp_v[j] * exp_scale * y_scale_inv) + y_zero;
      y = std::min(std::max(y, -128), 127);
      window_writeincr(out, y);
    }
    window_incr(in, INP_W_PAD - INP_W);
    window_incr(out, INP_W_PAD - INP_W);
  }

  PROFILE_FOOTER;
}

