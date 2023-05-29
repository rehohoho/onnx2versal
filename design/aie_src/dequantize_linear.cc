#include "dequantize_linear.h"
#include "kernel_utils.h"


template <int INP_SIZE, int OUT_SIZE>
void DequantizeLinearScalar<INP_SIZE, OUT_SIZE>::filter(
	input_window<int8_t>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running DequantizeLinearScalar<%d,%d>\n", INP_SIZE, OUT_SIZE));

  for (int i = 0; i < OUT_SIZE; i++) {
    int x = window_readincr(in);
    float y = (x - zero) * scale;
    window_writeincr(out, y);
  }

  PROFILE_FOOTER;
}
