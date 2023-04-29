#ifndef CONCAT_H_
#define CONCAT_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int L0, int L1, int L2, int L3>
void concat_scalar(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
);


#endif // CONCAT_H_
