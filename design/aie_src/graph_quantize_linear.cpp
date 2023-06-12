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
      int8_t y_zero
    ): g(y_scale, y_zero) { 
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

QuantizerLinearTest<QuantizeLinear, 28, 28, 32> quantizeLinearVector(
  "quantizeLinearVector", 
  "quantizelinear_int8in.txt", 
  "quantizelinear_int8out_shape28x28_QuantizeLinear.txt",
  0.00392156889, -128);

QuantizerLinearTest<QuantizeLinearFmul, 28, 28, 32> quantizeLinearFmul(
  "quantizeLinearFmul", 
  "quantizelinear_int8in.txt", 
  "quantizelinear_int8out_shape28x28_QuantizeLinearFmul.txt",
  0.00392156889, -128);

#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	adfCheck(quantizeLinearScalar.init(), "init quantizeLinearScalar");
  adfCheck(quantizeLinearScalar.run(ITER_CNT), "run quantizeLinearScalar");
	adfCheck(quantizeLinearScalar.end(), "end quantizeLinearScalar");

  adfCheck(quantizeLinearVector.init(), "init quantizeLinearVector");
  adfCheck(quantizeLinearVector.run(ITER_CNT), "run quantizeLinearVector");
	adfCheck(quantizeLinearVector.end(), "end quantizeLinearVector");

  adfCheck(quantizeLinearFmul.init(), "init quantizeLinearFmul");
  adfCheck(quantizeLinearFmul.run(ITER_CNT), "run quantizeLinearFmul");
	adfCheck(quantizeLinearFmul.end(), "end quantizeLinearFmul");
  return 0;
}
#endif
