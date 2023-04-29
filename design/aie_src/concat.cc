#include "concat.h"
#include "kernel_utils.h"


template <int NCHUNK, int OUTSIZE>
void concat4_scalar(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER;
  printf("Running concat4_scalar<%d, %d>", NCHUNK, OUTSIZE);
  int outOff = 0;

  for (int i = 0; i < NCHUNK; i++) {
    float a = window_readincr(in0);
    window_writeincr(out, a); outOff++;
    if (outOff >= OUTSIZE) break;
  }

  for (int i = 0; i < NCHUNK; i++) {
    float a = window_readincr(in1);
    window_writeincr(out, a); outOff++;
    if (outOff >= OUTSIZE) break;
  }

  for (int i = 0; i < NCHUNK; i++) {
    float a = window_readincr(in2);
    window_writeincr(out, a); outOff++;
    if (outOff >= OUTSIZE) break;
  }

  for (int i = 0; i < NCHUNK; i++) {
    float a = window_readincr(in3);
    window_writeincr(out, a); outOff++;
    if (outOff > OUTSIZE) break;
  }

  PROFILE_FOOTER;
}
