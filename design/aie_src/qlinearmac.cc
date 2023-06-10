#include "qlinearmac.h"
#include "kernel_utils.h"



template <int B, int W, int IS_RELU>
QlinearMacScalar<B, W, IS_RELU>::QlinearMacScalar (
  int8_t (&w)[W],
  int8_t (&b)[W],
  float x_scale,
  float w_scale,
  float b_scale,
  float z_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t b_zero,
  int8_t z_zero,
  int8_t y_zero
): 
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), b_scale(b_scale), z_scale(z_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), b_zero(b_zero), z_zero(z_zero), y_zero(y_zero) 
{
  scale_x = x_scale*w_scale/z_scale;
  scale_z = z_scale * inv(y_scale);
  for (int w = 0; w < W; w++) {
    shift_x[w] = -x_zero * (weights[w] - w_zero) * scale_x + z_zero;
    shift_z[w] = (-z_zero * z_scale + (bias[w] - b_zero) * b_scale) * inv(y_scale) + y_zero;
  }
}

template <int B, int W, int IS_RELU>
void QlinearMacScalar<B, W, IS_RELU>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QlinearMacScalar<%d,%d,%d>\n", B, W, IS_RELU));
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w++) {
      int x = window_readincr(in);
      x = x * (weights[w] - w_zero);
      x = round(x * scale_x + shift_x[w]);
      x = std::min(std::max(x, -128), 127);

      int y = round(x * scale_z + shift_z[w]);
      y = std::min(std::max(y, -128), 127);
      if (IS_RELU)
        y = (y > 0) ? y : 0;
      window_writeincr(out, y);
    }
  }

  PROFILE_FOOTER;
}
