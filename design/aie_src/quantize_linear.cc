#include "quantize_linear.h"
#include "kernel_utils.h"


template <int INP_H, int INP_W, int OUT_W>
void QuantizeLinearScalar<INP_H, INP_W, OUT_W>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearScalar<%d, %d,%d>\n", INP_H, INP_W, OUT_W));

  for (int i = 0; i < INP_H; i++) {
    for (int j = 0; j < INP_W; j++) {
      float x = window_readincr(in);
      int y = round(x / y_scale) + y_zero_point;
      y = std::min(std::max(y, -128), 128);
      window_writeincr(out, y);
    }
    window_incr(out, OUT_W-INP_W);
  }

  PROFILE_FOOTER;
}
