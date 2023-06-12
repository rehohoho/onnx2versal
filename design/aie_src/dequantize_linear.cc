#include "dequantize_linear.h"
#include "kernel_utils.h"


template <int B, int INP_W, int OUT_W>
void DequantizeLinearScalar<B, INP_W, OUT_W>::filter(
	input_window<int8_t>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running DequantizeLinearScalar<%d,%d,%d>\n", B, INP_W, OUT_W));

  for (int i = 0; i < B; i++) {
    for (int j = 0; j < OUT_W; j++) {
      int x = window_readincr(in);
      float y = (x - zero) * scale;
      window_writeincr(out, y);
    }
    window_incr(in, OUT_W - INP_W);
  }

  PROFILE_FOOTER;
}
