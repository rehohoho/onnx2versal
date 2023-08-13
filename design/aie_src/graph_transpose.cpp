#include "graph_transpose.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int, int> class TRANSPOSE, 
  typename TT, int B, int H, int W, int C, int PAD_W>
class TransposeGraphTest : public adf::graph {

  private:
    TransposeGraph<TRANSPOSE, TT, B, H, W, C, PAD_W> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    TransposeGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "transpose_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_transpose"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_transpose"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*H*PAD_W*C*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*H*W*C*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
// float32
TransposeGraphTest<TransposeScalarBHWC2BCHW, float_t, 1, 4, 4, 16, 4> fp_bhwc2bchw(
  "fp_bhwc2bchw", "transpose_fp_bchw_shape1x4x4x16.txt", "transpose_fp_bhwc_shape1x16x4x4_TransposeScalarBHWC2BCHW.txt");

TransposeGraphTest<TransposeScalarBCHW2BHWC, float_t, 1, 4, 4, 16, 4> fp_bchw2bhwc(
  "fp_bchw2bhwc", "transpose_fp_bhwc_shape1x16x4x4.txt", "transpose_fp_bchw_shape1x4x4x16_TransposeScalarBCHW2BHWC.txt");

// int8 or uint8
TransposeGraphTest<TransposeScalarBCHW2BHWC,uint8_t,1,8,8,32,16> k007transpose(
  "k007transpose",
  "k007transpose_in_shape1x32x8x16.txt",
  "k007transpose_goldenout_shape1x8x8x32.txt"
);

#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(fp_bhwc2bchw.init(), "init fp_bhwc2bchw");
  adfCheck(fp_bhwc2bchw.run(ITER_CNT), "run fp_bhwc2bchw");
	adfCheck(fp_bhwc2bchw.end(), "end fp_bhwc2bchw");

  adfCheck(fp_bchw2bhwc.init(), "init fp_bchw2bhwc");
  adfCheck(fp_bchw2bhwc.run(ITER_CNT), "run fp_bchw2bhwc");
	adfCheck(fp_bchw2bhwc.end(), "end fp_bchw2bhwc");

  adfCheck(k007transpose.init(), "init k007transpose");
  adfCheck(k007transpose.run(ITER_CNT), "run k007transpose");
	adfCheck(k007transpose.end(), "end k007transpose");
  return 0;
}
#endif
