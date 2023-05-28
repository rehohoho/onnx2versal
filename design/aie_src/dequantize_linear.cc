#include "dequantize_linear.h"
#include "kernel_utils.h"


template <int CHUNK_CNT, int CHUNK_SIZE, int CHUNK_SIZE_PAD>
void DequantizeLinearScalar<CHUNK_CNT, CHUNK_SIZE, CHUNK_SIZE_PAD>::filter(
	input_window<int8_t>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running DequantizeLinearScalar<%d,%d,%d>\n", CHUNK_CNT, CHUNK_SIZE, CHUNK_SIZE_PAD));

  for (int i = 0; i < CHUNK_CNT; i++) {
    for (int j = 0; j < CHUNK_SIZE; j++) {
      int x = window_readincr(in);
      float y = (x - zero) * scale;
      window_writeincr(out, y);
    }
    window_incr(in, CHUNK_SIZE_PAD - CHUNK_SIZE);
  }

  PROFILE_FOOTER;
}
