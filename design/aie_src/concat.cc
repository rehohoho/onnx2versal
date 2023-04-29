#include "concat.h"
#include "kernel_utils.h"


template <int L0, int L1, int L2, int L3, int OUTSIZE>
void concat_scalar(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER;
  printf("Running concat_scalar<%d, %d, %d, %d, %d>", L0, L1, L2, L3, OUTSIZE);
  int outOff = 0;

  for (int i = 0; i < L0; i++) {
    float a = window_readincr(in0);
    window_writeincr(out, a); outOff++;
    if (outOff >= OUTSIZE) break;
  }

  for (int i = 0; i < L1; i++) {
    float a = window_readincr(in1);
    window_writeincr(out, a); outOff++;
    if (outOff >= OUTSIZE) break;
  }

  for (int i = 0; i < L2; i++) {
    float a = window_readincr(in2);
    window_writeincr(out, a); outOff++;
    if (outOff >= OUTSIZE) break;
  }

  for (int i = 0; i < L3; i++) {
    float a = window_readincr(in3);
    window_writeincr(out, a); outOff++;
    if (outOff > OUTSIZE) break;
  }

  PROFILE_FOOTER;
}
