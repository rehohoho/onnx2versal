#ifndef SPLIT_H_
#define SPLIT_H_

#include <type_traits>
#include <adf.h>
#include <assert.h>


/** 
 * @defgroup SplitKernels
 * @ingroup Split
 * 
 * @{
 */


/**
 * @brief Scalar implementation, 
 * requires 2*OVERLAP <= OUT_W, (INP_W-OUT_W) % FIRST_STRIDE == 0
 * SplitScalar<f,3,10,64,22>::filter3 takes 9505 cycles
 */
template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
class SplitScalar {
	private:
		static constexpr int FIRST_STRIDE = OUT_W - OVERLAP;
		static constexpr int LCNT = (INP_W - OUT_W) / FIRST_STRIDE + 1;
		static constexpr int STRIDE = OUT_W - OVERLAP - OVERLAP;
	
	public:
		void filter8(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1,
			output_window<TT>* out2,
			output_window<TT>* out3,
			output_window<TT>* out4,
			output_window<TT>* out5,
			output_window<TT>* out6,
			output_window<TT>* out7
		);
		void filter7(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1,
			output_window<TT>* out2,
			output_window<TT>* out3,
			output_window<TT>* out4,
			output_window<TT>* out5,
			output_window<TT>* out6
		);
		void filter6(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1,
			output_window<TT>* out2,
			output_window<TT>* out3,
			output_window<TT>* out4,
			output_window<TT>* out5
		);
		void filter5(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1,
			output_window<TT>* out2,
			output_window<TT>* out3,
			output_window<TT>* out4
		);
		void filter4(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1,
			output_window<TT>* out2,
			output_window<TT>* out3
		);
		void filter3(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1,
			output_window<TT>* out2
		);
		void filter2(
			input_stream<TT>* in,
			output_window<TT>* out0,
			output_window<TT>* out1
		);
		void filter1(
			input_stream<TT>* in,
			output_window<TT>* out0
		);
		static void registerKernelClass() {
			static_assert(2*OVERLAP <= OUT_W && (INP_W-OUT_W) % FIRST_STRIDE == 0);
			if (LCNT == 8) {
				REGISTER_FUNCTION(SplitScalar::filter8);
			} else if (LCNT == 7) {
				REGISTER_FUNCTION(SplitScalar::filter7);
			} else if (LCNT == 6) {
				REGISTER_FUNCTION(SplitScalar::filter6);
			} else if (LCNT == 5) {
				REGISTER_FUNCTION(SplitScalar::filter5);
			} else if (LCNT == 4) {
				REGISTER_FUNCTION(SplitScalar::filter4);
			} else if (LCNT == 3) {
				REGISTER_FUNCTION(SplitScalar::filter3);
			} else if (LCNT == 2) {
				REGISTER_FUNCTION(SplitScalar::filter2);
			} else if (LCNT == 1) {
				REGISTER_FUNCTION(SplitScalar::filter1);
			}
		}
};
/** @}*/


#endif // SPLIT_H_
