#include "graph_dequantize_linear.h"
#include "graph_utils.h"


template <template<int> class DEQUANTIZE_LINEAR, int WINDOW_SIZE>
class DequantizeLinearScalarTest : public adf::graph {

  private:
    DequantizeLinearGraph<DEQUANTIZE_LINEAR, WINDOW_SIZE> g;
    static constexpr int PAD_WINSIZE = DequantizeLinearGraph<DEQUANTIZE_LINEAR, WINDOW_SIZE>::PAD_WINSIZE;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    DequantizeLinearScalarTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT,
      float y_scale,
      int y_zero_point
    ): g(y_scale, y_zero_point) { 
      plin[0] = adf::input_plio::create("plin0_dequantize_linear"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_dequantize_linear"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<PAD_WINSIZE>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<WINDOW_SIZE*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
DequantizeLinearScalarTest<DequantizeLinearScalar, 1*10> dequantizeLinearScalar(
  "dequantizeLinearScalar", 
  "k9dequantizeLinear_in.txt", 
  "k9dequantizeLinear_goldenout.txt",
  0.05811788, -128);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(dequantizeLinearScalar.init(), "init dequantizeLinearScalar");
  adfCheck(dequantizeLinearScalar.run(ITER_CNT), "run dequantizeLinearScalar");
	adfCheck(dequantizeLinearScalar.end(), "end dequantizeLinearScalar");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(dequantizeLinearScalar.init(), "init dequantizeLinearScalar");
  get_graph_throughput_by_port(dequantizeLinearScalar, "plout[0]", dequantizeLinearScalar.plout[0], 1*10, sizeof(float), ITER_CNT);
	adfCheck(dequantizeLinearScalar.end(), "end dequantizeLinearScalar");
  return 0;
}
#endif
