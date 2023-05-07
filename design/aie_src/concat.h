#ifndef CONCAT_H_
#define CONCAT_H_

#include <adf.h>
#include "aie_api/aie.hpp"


template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat8Scalar {
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
			REGISTER_FUNCTION(Concat8Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat7Scalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			input_window<float>* in6,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat7Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat6Scalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat6Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat5Scalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat5Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat4Scalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat4Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat3Scalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat3Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat2Scalar {
	public:
		void filter(
			input_window<float>* in0,
			input_window<float>* in1,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat2Scalar::filter);
		}
};

template <int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class Concat1Scalar {
	public:
		void filter(
			input_window<float>* in0,
			output_window<float>* out
		);
		static void registerKernelClass() {
			REGISTER_FUNCTION(Concat1Scalar::filter);
		}
};


#endif // CONCAT_H_
