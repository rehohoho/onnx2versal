#include "graph_quantize_linear.h"
#include "graph_utils.h"


template <template<typename, int, int, int> class QUANTIZE_LINEAR, int HCHUNK,
  typename TT, int INP_H, int INP_W, int OUT_W>
class QuantizeLinearChunkHPktStreamGraphTest : public adf::graph {

  private:
    QuantizeLinearChunkHPktStreamGraph<QUANTIZE_LINEAR, HCHUNK, TT, INP_H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QuantizeLinearChunkHPktStreamGraphTest(
      const std::string& id,
      float y_scale,
      TT y_zero,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ): g(y_scale, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_quantize_linear"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_quantize_linear"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};

// instance to be compiled and used in host within xclbin
// 1179 cycles for 7x stream VS 1750 cycles for single stream
// QuantizeLinearChunkHPktStreamGraphTest<QuantizeLinearFmulStream, 4, int8_t, 28, 28, 32> quantizeLinearFmulPktStream(
//   "quantizeLinearFmulPktStream", 
//   0.00392156889, -128,
//   "quantizelinear_int8in.txt", 
//   "quantizelinear_int8out_shape28x28_QuantizeLinearFmulPktStream.txt"
// );

float_t k000quantizelinear_scale = 0.018658448;
uint8_t k000quantizelinear_zero = 114;
QuantizeLinearChunkHPktStreamGraphTest<QuantizeLinearFmulStream,84,uint8_t,672,224,224> k000quantizelinear(
  "k000quantizelinear",
  k000quantizelinear_scale, k000quantizelinear_zero,
  "input.txt",
  "k000quantizelinear_goldenout_shape1x3x224x224.txt"
);
#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(k000quantizelinear.init(), "init k000quantizelinear");
  adfCheck(k000quantizelinear.run(ITER_CNT), "run k000quantizelinear");
	adfCheck(k000quantizelinear.end(), "end k000quantizelinear");

  // adfCheck(quantizeLinearFmulPktStream.init(), "init quantizeLinearFmulPktStream");
  // adfCheck(quantizeLinearFmulPktStream.run(ITER_CNT), "run quantizeLinearFmulPktStream");
	// adfCheck(quantizeLinearFmulPktStream.end(), "end quantizeLinearFmulPktStream");
  return 0;
}
#endif
