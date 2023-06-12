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
 * @brief Scalar implementation, ConcatScalar<5, 64, 16, 52> takes 650 cycles
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
			output_window<TT>* out
		);
		void filter7(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			output_window<TT>* out
		);
		void filter6(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			output_window<TT>* out
		);
		void filter5(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			output_window<TT>* out
		);
		void filter4(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			output_window<TT>* out
		);
		void filter3(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			output_window<TT>* out
		);
		void filter2(
			input_window<TT>* in0,
			input_window<TT>* in1,
			output_window<TT>* out
		);
		void filter1(
			input_window<TT>* in0,
			output_window<TT>* out
		);
		static void registerKernelClass() {
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
 * @brief Vector implementation, Requires INP_W%8=0, OUT_W%4=0.
 * ConcatFloat<5, 64, 16, 52> takes 232 cycles.
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
			output_window<TT>* out
		);
		void filter7(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			output_window<TT>* out
		);
		void filter6(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			output_window<TT>* out
		);
		void filter5(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			output_window<TT>* out
		);
		void filter4(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			output_window<TT>* out
		);
		void filter3(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			output_window<TT>* out
		);
		void filter2(
			input_window<TT>* in0,
			input_window<TT>* in1,
			output_window<TT>* out
		);
		void filter1(
			input_window<TT>* in0,
			output_window<TT>* out
		);
		static void registerKernelClass() {
			static_assert(INP_W%8==0 && OUT_W%4==0 && (std::is_same<TT, float>::value));
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
 * requires INP_W%16=0, OUT_W%16=0.
 * ConcatInt8<5, 64, 16, 52> takes 223 cycles.
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
			output_window<TT>* out
		);
		void filter7(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			input_window<TT>* in6,
			output_window<TT>* out
		);
		void filter6(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			input_window<TT>* in5,
			output_window<TT>* out
		);
		void filter5(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			input_window<TT>* in4,
			output_window<TT>* out
		);
		void filter4(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			input_window<TT>* in3,
			output_window<TT>* out
		);
		void filter3(
			input_window<TT>* in0,
			input_window<TT>* in1,
			input_window<TT>* in2,
			output_window<TT>* out
		);
		void filter2(
			input_window<TT>* in0,
			input_window<TT>* in1,
			output_window<TT>* out
		);
		void filter1(
			input_window<TT>* in0,
			output_window<TT>* out
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
/** @}*/


#endif // CONCAT_H_
