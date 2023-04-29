#ifndef CONCAT_H_
#define CONCAT_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int NCHUNK, int OUTSIZE>
void concat4_scalar(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
);


#endif // CONCAT_H_
