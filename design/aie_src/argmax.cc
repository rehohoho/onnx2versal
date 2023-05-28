#include "argmax.h"
#include "kernel_utils.h"


template <int CHUNK_CNT, int CHUNK_SIZE>
void ArgmaxScalar<CHUNK_CNT, CHUNK_SIZE>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ArgmaxScalar<%d,%d>\n", CHUNK_CNT, CHUNK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    float c = -std::numeric_limits<double>::infinity();
    int cidx = -1;
    for (int j = 0; j < CHUNK_SIZE; j++) {
      float a = window_readincr(in);
      if (a > c) {
        c = a;
        cidx = j;
      }
    }
    window_writeincr(out, cidx);
  }

  PROFILE_FOOTER;
}