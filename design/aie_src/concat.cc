#include "concat.h"
#include "kernel_utils.h"


#define CAT(INP_WIN) \
  for (int i = 0; i < NCHUNK; i++) { \
    if (outOff >= OUTSIZE) break; \
    float a = window_readincr(INP_WIN); \
    window_writeincr(out, a); outOff++;}


template <int NCHUNK, int OUTSIZE>
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
  printf("Running concat8_scalar<%d, %d>", NCHUNK, OUTSIZE);
  int outOff = 0;

  CAT(in0);
  CAT(in1);
  CAT(in2);
  CAT(in3);
  CAT(in4);
  CAT(in5);
  CAT(in6);
  CAT(in7);

  PROFILE_FOOTER;
}
