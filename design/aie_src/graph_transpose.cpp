#include "graph_transpose.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class TRANSPOSE, 
  typename TT, int B, int H, int W, int C>
class TransposeGraphTest : public adf::graph {

  private:
    TransposeGraph<TRANSPOSE, TT, B, H, W, C> g;

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
      adf::connect<adf::window<B*H*W*C*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
TransposeGraphTest<TransposeScalarBHWC2BCHW, float_t, 1, 4, 4, 16> fpscalar(
  "fpscalar", "transpose_fpin.txt", "transpose_fpout_shape1x16x4x4_TransposeScalarBHWC2BCHW.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(fpscalar.init(), "init fpscalar");
  adfCheck(fpscalar.run(ITER_CNT), "run fpscalar");
	adfCheck(fpscalar.end(), "end fpscalar");
  return 0;
}
#endif
