#ifndef CONCAT_H_
#define CONCAT_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int NLANES, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class ConcatScalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			input_window<float>* in6,
			input_window<float>* in7,
			output_window<float>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(ConcatScalar::filter);
		}
};


#endif // CONCAT_H_
