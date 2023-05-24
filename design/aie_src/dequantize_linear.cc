#include "dequantize_linear.h"
#include "kernel_utils.h"


template <int WINDOW_SIZE>
void DequantizeLinearScalar<WINDOW_SIZE>::filter(
	input_window<int8_t>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running DequantizeLinearScalar<%d>\n", WINDOW_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i++) {
    int x = window_readincr(in);
    float y = (x - zero) * scale;
    window_writeincr(out, y);
  }

  PROFILE_FOOTER;
}
