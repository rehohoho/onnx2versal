#include "graph_qlinearadd.h"
#include "graph_utils.h"


template <template<typename, int, int> class QLINEARADD, 
  typename TT, int W, int IS_RELU>
class QLinearAddGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    QLinearAddGraph<QLINEARADD, TT, W, IS_RELU> g;

  public:
    adf::input_plio plin[2];
    adf::output_plio plout[1];

    QLinearAddGraphTest(
      const std::string& id,
      float a_scale,
			float b_scale,
			float c_scale,
			TT a_zero,
			TT b_zero,
			TT c_zero,
      const std::string& INPA_TXT, 
      const std::string& INPB_TXT, 
      const std::string& OUT_TXT
    ): g(a_scale, b_scale, c_scale, a_zero, b_zero, c_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearadd"+id, PLIO64_ARG(INPA_TXT));
      plin[1] = adf::input_plio::create("plin1_qlinearadd"+id, PLIO64_ARG(INPB_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearadd"+id, PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (plin[1].out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
QLinearAddGraphTest<QLinearAddInt8, int8_t, 16384, 0> qlinearaddScalar(
  "qlinearaddScalar", 0.036632415, 0.11731018, 0.062222864, -128, -6, -128,
  "k005qlinearadd_inA_shape1x16x32x32.txt", 
  "k005qlinearadd_inB_shape1x16x32x32.txt",
  "k005qlinearadd_goldenout_shape1x16x32x32_scalar.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	adfCheck(qlinearaddScalar.init(), "init qlinearaddScalar");
  adfCheck(qlinearaddScalar.run(ITER_CNT), "run qlinearaddScalar");
	adfCheck(qlinearaddScalar.end(), "end qlinearaddScalar");
  return 0;
}
#endif
