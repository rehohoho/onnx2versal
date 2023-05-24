#include "graph_quantize_linear.h"
#include "graph_utils.h"


template <template<int> class QUANTIZE_LINEAR, int WINDOW_SIZE>
class QuantizeLinearScalarTest : public adf::graph {

  private:
    QuantizeLinearGraph<QUANTIZE_LINEAR, WINDOW_SIZE> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QuantizeLinearScalarTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT,
      float y_scale,
      int y_zero_point
    ): g(y_scale, y_zero_point) { 
      plin[0] = adf::input_plio::create("plin0_quantize_linear"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_quantize_linear"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<WINDOW_SIZE*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<WINDOW_SIZE>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
QuantizeLinearScalarTest<QuantizeLinearScalar, 1*1*28*28> quantizeLinearScalar(
  "quantizeLinearScalar", 
  "k0quantizelinear_in.txt", 
  "k0quantizelinear_goldenout.txt",
  3.921568859368562698e-03, -128);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(quantizeLinearScalar.init(), "init quantizeLinearScalar");
  adfCheck(quantizeLinearScalar.run(ITER_CNT), "run quantizeLinearScalar");
	adfCheck(quantizeLinearScalar.end(), "end quantizeLinearScalar");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(quantizeLinearScalar.init(), "init quantizeLinearScalar");
  get_graph_throughput_by_port(quantizeLinearScalar, "plout[0]", quantizeLinearScalar.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(quantizeLinearScalar.end(), "end quantizeLinearScalar");
  return 0;
}
#endif
