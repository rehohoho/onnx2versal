#include "concat.h"
#include "kernel_utils.h"


#define CAT(INP_WIN) \
  for (int i = 0; i < NCHUNK; i++) { \
    if (outOff >= OUTSIZE) break; \
    float a = window_readincr(INP_WIN); \
    window_writeincr(out, a); outOff++;}


template <int NCHUNK, int OUTSIZE>
void concat32_scalar(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
	input_window<float>* in6,
	input_window<float>* in7,
	input_window<float>* in8,
	input_window<float>* in9,
	input_window<float>* in10,
	input_window<float>* in11,
	input_window<float>* in12,
	input_window<float>* in13,
	input_window<float>* in14,
	input_window<float>* in15,
	input_window<float>* in16,
	input_window<float>* in17,
	input_window<float>* in18,
	input_window<float>* in19,
	input_window<float>* in20,
	input_window<float>* in21,
	input_window<float>* in22,
	input_window<float>* in23,
	input_window<float>* in24,
	input_window<float>* in25,
	input_window<float>* in26,
	input_window<float>* in27,
	input_window<float>* in28,
	input_window<float>* in29,
	input_window<float>* in30,
	input_window<float>* in31,
  output_window<float>* out
) {
  PROFILE_HEADER;
  printf("Running concat32_scalar<%d, %d>", NCHUNK, OUTSIZE);
  uint outOff = 0;

  CAT(in0);
  CAT(in1);
  CAT(in2);
  CAT(in3);
  CAT(in4);
  CAT(in5);
  CAT(in6);
  CAT(in7);
  CAT(in8);
  CAT(in9);
  CAT(in10);
  CAT(in11);
  CAT(in12);
  CAT(in13);
  CAT(in14);
  CAT(in15);
  CAT(in16);
  CAT(in17);
  CAT(in18);
  CAT(in19);
  CAT(in20);
  CAT(in21);
  CAT(in22);
  CAT(in23);
  CAT(in24);
  CAT(in25);
  CAT(in26);
  CAT(in27);
  CAT(in28);
  CAT(in29);
  CAT(in30);
  CAT(in31);

  PROFILE_FOOTER;
}
