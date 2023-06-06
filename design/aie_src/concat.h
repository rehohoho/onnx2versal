#ifndef CONCAT_H_
#define CONCAT_H_

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
template <int LCNT, int H, int INP_W, int OUT_W>
class ConcatScalar {
	public:
		void filter8(
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
		void filter7(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			input_window<float>* in6,
			output_window<float>* out
		);
		void filter6(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			output_window<float>* out
		);
		void filter5(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			output_window<float>* out
		);
		void filter4(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			output_window<float>* out
		);
		void filter3(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			output_window<float>* out
		);
		void filter2(
			input_window<float>* in0,
			input_window<float>* in1,
			output_window<float>* out
		);
		void filter1(
			input_window<float>* in0,
			output_window<float>* out
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
 * ConcatVector<5, 64, 16, 52> takes 232 cycles.
 */
template <int LCNT, int H, int INP_W, int OUT_W>
class ConcatVector {
	public:
		void filter8(
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
		void filter7(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			input_window<float>* in6,
			output_window<float>* out
		);
		void filter6(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			input_window<float>* in5,
			output_window<float>* out
		);
		void filter5(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			input_window<float>* in4,
			output_window<float>* out
		);
		void filter4(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			input_window<float>* in3,
			output_window<float>* out
		);
		void filter3(
			input_window<float>* in0,
			input_window<float>* in1,
			input_window<float>* in2,
			output_window<float>* out
		);
		void filter2(
			input_window<float>* in0,
			input_window<float>* in1,
			output_window<float>* out
		);
		void filter1(
			input_window<float>* in0,
			output_window<float>* out
		);
		static void registerKernelClass() {
			static_assert(INP_W%8==0 && OUT_W%4==0);
			if (LCNT == 8) {
				REGISTER_FUNCTION(ConcatVector::filter8);
			} else if (LCNT == 7) {
				REGISTER_FUNCTION(ConcatVector::filter7);
			} else if (LCNT == 6) {
				REGISTER_FUNCTION(ConcatVector::filter6);
			} else if (LCNT == 5) {
				REGISTER_FUNCTION(ConcatVector::filter5);
			} else if (LCNT == 4) {
				REGISTER_FUNCTION(ConcatVector::filter4);
			} else if (LCNT == 3) {
				REGISTER_FUNCTION(ConcatVector::filter3);
			} else if (LCNT == 2) {
				REGISTER_FUNCTION(ConcatVector::filter2);
			} else if (LCNT == 1) {
				REGISTER_FUNCTION(ConcatVector::filter1);
			}
		}
};
/** @}*/


#endif // CONCAT_H_
