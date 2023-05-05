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
char transpose1_input[]  = "lenet_mnist__13___Reshape___pool2_MaxPool_output_0__1x4x4x16.txt";
char transpose1_output[] = "lenet_mnist__13___Reshape___Reshape_output_0__1x256.txt";
TransposeGraphTest<TransposeScalarBHWC2BCHW, 1, 4, 4, 16> transpose1(
  "1", transpose1_input, transpose1_output);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	
  adfCheck(transpose1.init(), "init transpose1");
  adfCheck(transpose1.run(1), "run transpose1");
	adfCheck(transpose1.end(), "end transpose1");

  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(transpose1.init(), "init transpose1");
  get_graph_throughput_by_port(transpose1, "plout[0]", transpose1.plout[0], 1*4*4*16, sizeof(float32), ITER_CNT);
	adfCheck(transpose1.end(), "end transpose1");

  return 0;
}
#endif
