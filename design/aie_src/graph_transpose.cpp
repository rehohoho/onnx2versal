#include "graph_transpose.h"
#include "graph_utils.h"

#define ITER_CNT 1


template <template<int, int, int, int> class TRANSPOSE, 
  int B, int H, int W, int C>
class TransposeGraphTest : public adf::graph {

  private:
    TransposeGraph<TRANSPOSE, B, H, W, C> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    TransposeGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "transpose_out.txt"
    ) { 
      g.construct();
      plin[0] = adf::input_plio::create("plin0_transpose"+id+"_input", adf::plio_64_bits, TXT_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_transpose"+id+"_output", adf::plio_64_bits, TXT_ARG(OUT_TXT));
      adf::connect<adf::window<B*H*W*C*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*C*H*W*4>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
TransposeGraphTest<TransposeScalarBHWC2BCHW, 1, 4, 4, 16> scalar(
  "scalar", "transpose_fpin.txt", "transpose_fpout_TransposeScalarBHWC2BCHW.txt");
TransposeGraphTest<TransposeScalarBHWC2BCHW, 1, 4, 4, 16> scalar_rand(
  "scalar_rand", "transpose_fpin.txt", "transpose_fpout_TransposeScalarBHWC2BCHW_rand.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(scalar.init(), "init scalar");
  adfCheck(scalar.run(1), "run scalar");
	adfCheck(scalar.end(), "end scalar");

  adfCheck(scalar_rand.init(), "init scalar_rand");
  adfCheck(scalar_rand.run(1), "run scalar_rand");
	adfCheck(scalar_rand.end(), "end scalar_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  adfCheck(scalar.init(), "init scalar");
  get_graph_throughput_by_port(scalar, "plout[0]", scalar.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(scalar.end(), "end scalar");

  adfCheck(scalar_rand.init(), "init scalar_rand");
  get_graph_throughput_by_port(scalar_rand, "plout[0]", scalar_rand.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(scalar_rand.end(), "end scalar_rand");
  return 0;
}
#endif
