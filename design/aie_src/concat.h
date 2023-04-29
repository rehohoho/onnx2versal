#ifndef CONCAT_H_
#define CONCAT_H_

#include <adf.h>
#include "aie_api/aie.hpp"


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
);


#endif // CONCAT_H_
