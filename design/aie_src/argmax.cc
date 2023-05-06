#include "argmax.h"
#include "kernel_utils.h"


template <int WINDOW_SIZE, int CHUNK_SIZE>
void ArgmaxScalar<WINDOW_SIZE, CHUNK_SIZE>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER;
  printf("Running ArgmaxScalar<%d, %d>\n", WINDOW_SIZE, CHUNK_SIZE);

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
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