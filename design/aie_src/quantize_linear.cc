#include "quantize_linear.h"
#include "kernel_utils.h"


template <int WINDOW_SIZE>
void QuantizeLinearScalar<WINDOW_SIZE>::filter(
	input_window<float>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QuantizeLinearScalar<%d>\n", WINDOW_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i++) {
    float x = window_readincr(in);
    int y = round(x / y_scale) + y_zero_point;
    y = std::min(std::max(y, -128), 128);
    window_writeincr(out, y);
  }

  PROFILE_FOOTER;
}
