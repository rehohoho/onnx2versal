#include "graph_pad.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int, int, int, int> class PAD, 
  typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
class Pad2DGraphTest : public adf::graph {

  private:
    Pad2DGraph<PAD, TT, B, INP_H, INP_W, H0, H1, W0, W1> g;
    static constexpr int OUT_H = INP_H + H0 + H1;
    static constexpr int OUT_W = INP_W + W0 + W1;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    Pad2DGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "pad_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_pad_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_pad_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
const int B = 2;
const int INP_H = 32;
const int INP_W = 32;
const int H0 = 1;
const int H1 = 1;
const int W0 = 1;
const int W1 = 1;

Pad2DGraphTest<Pad2DScalar, float_t, B, INP_H, INP_W, H0, H1, W0, W1> pad2DScalar(
  "pad2DScalar", "pad_2d_fpin.txt", "pad_2d_fpout_shape2x34x34_Pad2DScalar.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(pad2DScalar.init(), "init pad2DScalar");
  adfCheck(pad2DScalar.run(ITER_CNT), "run pad2DScalar");
	adfCheck(pad2DScalar.end(), "end pad2DScalar");
  return 0;
}
#endif
