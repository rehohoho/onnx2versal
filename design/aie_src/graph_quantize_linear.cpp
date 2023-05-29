#include "graph_quantize_linear.h"
#include "graph_utils.h"


template <template<int, int, int> class QUANTIZE_LINEAR, int INP_H, int INP_W, int OUT_W>
class QuantizerLinearTest : public adf::graph {

  private:
    QuantizeLinearGraph<QUANTIZE_LINEAR, INP_H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QuantizerLinearTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT,
      float y_scale,
      int y_zero_point
    ): g(y_scale, y_zero_point) { 
      plin[0] = adf::input_plio::create("plin0_quantize_linear"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_quantize_linear"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<INP_H*INP_W*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<INP_H*OUT_W>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
QuantizerLinearTest<QuantizeLinearScalar, 28, 28, 32> quantizeLinearScalar(
  "quantizeLinearScalar", 
  "quantizelinear_int8in.txt", 
  "quantizelinear_int8out_shape28x28_QuantizeLinearScalar.txt",
  0.00392156889, -128);

QuantizerLinearTest<QuantizeLinearVector, 28, 28, 32> quantizeLinearVector(
  "quantizeLinearVector", 
  "quantizelinear_int8in.txt", 
  "quantizelinear_int8out_shape28x28_QuantizeLinearVector.txt",
  0.00392156889, -128);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(quantizeLinearScalar.init(), "init quantizeLinearScalar");
  adfCheck(quantizeLinearScalar.run(ITER_CNT), "run quantizeLinearScalar");
	adfCheck(quantizeLinearScalar.end(), "end quantizeLinearScalar");

  adfCheck(quantizeLinearVector.init(), "init quantizeLinearVector");
  adfCheck(quantizeLinearVector.run(ITER_CNT), "run quantizeLinearVector");
	adfCheck(quantizeLinearVector.end(), "end quantizeLinearVector");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(quantizeLinearScalar.init(), "init quantizeLinearScalar");
  get_graph_throughput_by_port(quantizeLinearScalar, "plout[0]", quantizeLinearScalar.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(quantizeLinearScalar.end(), "end quantizeLinearScalar");

  adfCheck(quantizeLinearVector.init(), "init quantizeLinearVector");
  get_graph_throughput_by_port(quantizeLinearVector, "plout[0]", quantizeLinearVector.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(quantizeLinearVector.end(), "end quantizeLinearVector");
  return 0;
}
#endif
