#include "concat.h"
#include "kernel_utils.h"


#define CAT(INP_WIN, CHUNK_SIZE) \
  for (int i = 0; i < CHUNK_SIZE; i++) { \
    if (outOff >= OUTSIZE) break; \
    float a = window_readincr(INP_WIN); \
    window_writeincr(out, a); outOff++;}


/*
Assumes WINDOW SIZE is divisible by CHUNKSIZE
lenet fc1: 294 cycles
lenet fc2: 349 cycles
lenet fc3: 91 cycles
*/
template <int NLANES, int N, int CHUNKSIZE, int OUTSIZE>
void concat8_scalar(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
	input_window<float>* in6,
	input_window<float>* in7,
  output_window<float>* out
) {
  PROFILE_HEADER;
  printf("Running concat8_scalar<%d, %d, %d>\n", N, CHUNKSIZE, OUTSIZE);
  int outOff = 0;

  for (int i = 0; i < N; i+=CHUNKSIZE) {
    if (NLANES >= 1) CAT(in0, CHUNKSIZE);
    if (NLANES >= 2) CAT(in1, CHUNKSIZE);
    if (NLANES >= 3) CAT(in2, CHUNKSIZE);
    if (NLANES >= 4) CAT(in3, CHUNKSIZE);
    if (NLANES >= 5) CAT(in4, CHUNKSIZE);
    if (NLANES >= 6) CAT(in5, CHUNKSIZE);
    if (NLANES >= 7) CAT(in6, CHUNKSIZE);
    if (NLANES >= 8) CAT(in7, CHUNKSIZE);
  }

  PROFILE_FOOTER;
}