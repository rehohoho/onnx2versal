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
 * @brief Scalar implementation for 32-bit stream, 
 * requires 2*OVERLAP <= OUT_W, (INP_W-OUT_W) % FIRST_STRIDE == 0
 * SplitScalar<f,3,10,64,22>::filter3 takes 9505 cycles
 * SplitScalar<f,10,64,31,-1>::filter2 takes 9547 cycles
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
			static_assert((OVERLAP < 0) || (2*OVERLAP <= OUT_W && (INP_W-OUT_W) % FIRST_STRIDE == 0));
			static_assert((OVERLAP > 0) || (OUT_W*LCNT - OVERLAP*(LCNT-1) <= INP_W));
			static_assert(sizeof(TT) == 4); // 32-bit width stream
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


/**
 * @brief Scalar implementation for int8 stream, 
 * requires 2*OVERLAP <= OUT_W, (INP_W-OUT_W) % FIRST_STRIDE == 0
 * SplitInt8<a,10,160,64,16>::filter3 takes 1649 cycles
 * SplitInt8<a,10,160,64,-32>::filter2 takes 1237 cycles
 */
template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
class SplitInt8 {
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
			static_assert((std::is_same<TT, int8_t>::value));
			static_assert((OVERLAP < 0) || (FIRST_STRIDE%16==0 && OVERLAP%16==0 && STRIDE%16==0));
			static_assert((OVERLAP < 0) || (2*OVERLAP <= OUT_W && (INP_W-OUT_W) % FIRST_STRIDE == 0));

			static_assert((OVERLAP > 0) || (OUT_W*LCNT - OVERLAP*(LCNT-1) <= INP_W));
			static_assert((OVERLAP > 0) || (OUT_W%16==0 && OVERLAP%16==0));

			if (LCNT == 8) {
				REGISTER_FUNCTION(SplitInt8::filter8);
			} else if (LCNT == 7) {
				REGISTER_FUNCTION(SplitInt8::filter7);
			} else if (LCNT == 6) {
				REGISTER_FUNCTION(SplitInt8::filter6);
			} else if (LCNT == 5) {
				REGISTER_FUNCTION(SplitInt8::filter5);
			} else if (LCNT == 4) {
				REGISTER_FUNCTION(SplitInt8::filter4);
			} else if (LCNT == 3) {
				REGISTER_FUNCTION(SplitInt8::filter3);
			} else if (LCNT == 2) {
				REGISTER_FUNCTION(SplitInt8::filter2);
			} else if (LCNT == 1) {
				REGISTER_FUNCTION(SplitInt8::filter1);
			}
		}
};
/** @}*/


#endif // SPLIT_H_
