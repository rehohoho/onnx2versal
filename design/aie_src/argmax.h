#ifndef ARGMAX_H_
#define ARGMAX_H_

#include <adf.h>


template <int WINDOW_SIZE, int CHUNK_SIZE>
class ArgmaxScalar {
	public:
		void filter(
			input_window<float>* in,
			output_window<float>* out
		);

		static void registerKernelClass() {
			REGISTER_FUNCTION(ArgmaxScalar::filter);
		}
};


#endif // ARGMAX_H_
