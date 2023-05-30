#include "softmax.h"
#include "kernel_utils.h"


template <int CHUNK_COUNT, int CHUNK_SIZE>
void SoftmaxScalar<CHUNK_COUNT, CHUNK_SIZE>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running SoftmaxScalar<%d,%d>\n", CHUNK_COUNT, CHUNK_SIZE));

  float exp_v[CHUNK_SIZE];
  float exp_scale;

  for (int i = 0; i < CHUNK_COUNT; i++) {
    exp_scale = 0;
    for (int j = 0; j < CHUNK_SIZE; j++) {
      float a = window_readincr(in);
      exp_v[j] = expf(a);
      exp_scale += exp_v[j];
    }
    exp_scale = 1 / exp_scale;
    for (int j = 0; j < CHUNK_SIZE; j++) {
      window_writeincr(out, exp_v[j] * exp_scale);
    }
  }

  PROFILE_FOOTER;
}