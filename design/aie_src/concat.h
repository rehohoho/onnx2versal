#ifndef CONCAT_H_
#define CONCAT_H_

#include <type_traits>
#include <adf.h>
#include <assert.h>


/** 
 * @defgroup ConcatKernels
 * @ingroup Concat
 * 
 * @{
 */


/**
 * @brief Scalar implementation, 
 * ConcatScalar<f,5,4,32,144> takes 5858 cycles (~850 for output window)
 */
template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatScalar {
	public:
		void filter8(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			input_window<TT>* in7,
			output_stream<TT>* out
		);
		void filter7(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			output_stream<TT>* out
		);
		void filter6(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			output_stream<TT>* out
		);
		void filter5(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			output_stream<TT>* out
		);
		void filter4(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			output_stream<TT>* out
		);
		void filter3(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			output_stream<TT>* out
		);
		void filter2(
			input_window<TT>* in0,
			input_window<TT>* in1,
			output_stream<TT>* out
		);
		void filter1(
			input_window<TT>* in0,
			output_stream<TT>* out
		);
		static void registerKernelClass() {
			static_assert(sizeof(TT) == 4); // 32-bit stream
			if (LCNT == 8) {
				REGISTER_FUNCTION(ConcatScalar::filter8);
			} else if (LCNT == 7) {
				REGISTER_FUNCTION(ConcatScalar::filter7);
			} else if (LCNT == 6) {
				REGISTER_FUNCTION(ConcatScalar::filter6);
			} else if (LCNT == 5) {
				REGISTER_FUNCTION(ConcatScalar::filter5);
			} else if (LCNT == 4) {
				REGISTER_FUNCTION(ConcatScalar::filter4);
			} else if (LCNT == 3) {
				REGISTER_FUNCTION(ConcatScalar::filter3);
			} else if (LCNT == 2) {
				REGISTER_FUNCTION(ConcatScalar::filter2);
			} else if (LCNT == 1) {
				REGISTER_FUNCTION(ConcatScalar::filter1);
			}
		}
};


/**
 * @brief Vector implementation, 
 * requires INP_W%4=0, OUT_W%4=0.
 * ConcatFloat<f,5,4,32,144> takes 715 cycles (~300 for output window).
 */
template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatFloat {
	public:
		void filter8(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			input_window<TT>* in7,
			output_stream<TT>* out
		);
		void filter7(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			output_stream<TT>* out
		);
		void filter6(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			output_stream<TT>* out
		);
		void filter5(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			output_stream<TT>* out
		);
		void filter4(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			output_stream<TT>* out
		);
		void filter3(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			output_stream<TT>* out
		);
		void filter2(
			input_window<TT>* in0,
			input_window<TT>* in1,
			output_stream<TT>* out
		);
		void filter1(
			input_window<TT>* in0,
			output_stream<TT>* out
		);
		static void registerKernelClass() {
			static_assert(INP_W%4==0 && OUT_W%4==0 && (std::is_same<TT, float>::value));
			if (LCNT == 8) {
				REGISTER_FUNCTION(ConcatFloat::filter8);
			} else if (LCNT == 7) {
				REGISTER_FUNCTION(ConcatFloat::filter7);
			} else if (LCNT == 6) {
				REGISTER_FUNCTION(ConcatFloat::filter6);
			} else if (LCNT == 5) {
				REGISTER_FUNCTION(ConcatFloat::filter5);
			} else if (LCNT == 4) {
				REGISTER_FUNCTION(ConcatFloat::filter4);
			} else if (LCNT == 3) {
				REGISTER_FUNCTION(ConcatFloat::filter3);
			} else if (LCNT == 2) {
				REGISTER_FUNCTION(ConcatFloat::filter2);
			} else if (LCNT == 1) {
				REGISTER_FUNCTION(ConcatFloat::filter1);
			}
		}
};


/**
 * @brief Vector implementation for int8_t, 
 * requires INP_W%16=0, OUT_W%16=0,
 * ConcatInt8<f,5,4,32,144> takes 283 cycles (~same with output window).
 */
template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatInt8 {
	public:
		void filter8(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			input_window<TT>* in7,
			output_stream<TT>* out
		);
		void filter7(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			output_stream<TT>* out
		);
		void filter6(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			output_stream<TT>* out
		);
		void filter5(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			output_stream<TT>* out
		);
		void filter4(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			output_stream<TT>* out
		);
		void filter3(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			output_stream<TT>* out
		);
		void filter2(
			input_window<TT>* in0,
			input_window<TT>* in1,
			output_stream<TT>* out
		);
		void filter1(
			input_window<TT>* in0,
			output_stream<TT>* out
		);
		static void registerKernelClass() {
			static_assert(INP_W%16==0 && OUT_W%16==0 && (std::is_same<TT, int8_t>::value));
			if (LCNT == 8) {
				REGISTER_FUNCTION(ConcatInt8::filter8);
			} else if (LCNT == 7) {
				REGISTER_FUNCTION(ConcatInt8::filter7);
			} else if (LCNT == 6) {
				REGISTER_FUNCTION(ConcatInt8::filter6);
			} else if (LCNT == 5) {
				REGISTER_FUNCTION(ConcatInt8::filter5);
			} else if (LCNT == 4) {
				REGISTER_FUNCTION(ConcatInt8::filter4);
			} else if (LCNT == 3) {
				REGISTER_FUNCTION(ConcatInt8::filter3);
			} else if (LCNT == 2) {
				REGISTER_FUNCTION(ConcatInt8::filter2);
			} else if (LCNT == 1) {
				REGISTER_FUNCTION(ConcatInt8::filter1);
			}
		}
};


/**
 * @brief Scalar implementation for stream concat, 
 * ConcatScalarStream<f,4,32,32,64> takes ~1000 cycles
 */
template <typename TT, int H, int INP_W1, int INP_W2, int OUT_W>
class ConcatScalarStream {
	public:
		void filter(
			input_stream<TT>* in0,
			input_stream<TT>* in1,
			output_stream<TT>* out
		);
		static void registerKernelClass() {
			static_assert(sizeof(TT) == 4); // 32-bit stream
			// also expects INP_W1 < OUT_W, not included due to conditional instances in graph
			REGISTER_FUNCTION(ConcatScalarStream::filter);
		}
};


/**
 * @brief Vector implementation for stream concat with int8, 
 * ConcatInt8Stream<f,4,32,32,64> takes  cycles
 */
template <typename TT, int H, int INP_W1, int INP_W2, int OUT_W>
class ConcatInt8Stream {
	public:
		void filter(
			input_stream<TT>* in0,
			input_stream<TT>* in1,
			output_stream<TT>* out
		);
		static void registerKernelClass() {
			static_assert(INP_W1 % 16 == 0 && INP_W2 % 16 == 0 && OUT_W % 16 == 0);
			static_assert((std::is_same<TT, int8_t>::value));
			// also expects INP_W1 < OUT_W, not included due to conditional instances in graph
			REGISTER_FUNCTION(ConcatInt8Stream::filter);
		}
};
/** @}*/


#endif // CONCAT_H_
