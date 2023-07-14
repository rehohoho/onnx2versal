#include "graph_dequantize_linear.h"
#include "graph_utils.h"


template <template<typename, int, int, int> class DEQUANTIZE_LINEAR, 
  typename TT, int B, int INP_W, int OUT_W>
class DequantizeLinearScalarTest : public adf::graph {

  private:
    DequantizeLinearGraph<DEQUANTIZE_LINEAR, TT, B, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    DequantizeLinearScalarTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT,
      float y_scale,
      TT y_zero_point
    ): g(y_scale, y_zero_point) { 
      plin[0] = adf::input_plio::create("plin0_dequantize_linear"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_dequantize_linear"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_W>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_W*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
typedef int8_t TT;
const int B = 1;
const int INP_W = 96;
const int OUT_W = 84;
DequantizeLinearScalarTest<DequantizeLinearScalar, TT, B, INP_W, OUT_W> dequantizeLinearScalar(
  "dequantizeLinearScalar", 
  "dequantizelinear_int8in.txt", 
  "dequantizelinear_fpout_shape84_DequantizeLinearScalar.txt",
  0.00392156889, -128);

DequantizeLinearScalarTest<DequantizeLinear, TT, B, INP_W, OUT_W> dequantizeLinear(
  "dequantizeLinear", 
  "dequantizelinear_int8in.txt", 
  "dequantizelinear_fpout_shape84_DequantizeLinear.txt",
  0.00392156889, -128);


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	adfCheck(dequantizeLinearScalar.init(), "init dequantizeLinearScalar");
  adfCheck(dequantizeLinearScalar.run(ITER_CNT), "run dequantizeLinearScalar");
	adfCheck(dequantizeLinearScalar.end(), "end dequantizeLinearScalar");

  adfCheck(dequantizeLinear.init(), "init dequantizeLinear");
  adfCheck(dequantizeLinear.run(ITER_CNT), "run dequantizeLinear");
	adfCheck(dequantizeLinear.end(), "end dequantizeLinear");
  return 0;
}
#endif
