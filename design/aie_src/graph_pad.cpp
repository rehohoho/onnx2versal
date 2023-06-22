#include "graph_pad.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int, int, int, int> class PAD, 
  typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
class Pad2DStreamGraphTest : public adf::graph {

  private:
    Pad2DStreamGraph<PAD, TT, B, INP_H, INP_W, H0, H1, W0, W1> g;
    static constexpr int OUT_H = INP_H + H0 + H1;
    static constexpr int OUT_W = INP_W + W0 + W1;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    Pad2DStreamGraphTest(
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

template <template<typename, int, int, int, int, int, int, int> class PAD, 
  typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
class Pad2DWindowGraphTest : public adf::graph {

  private:
    Pad2DWindowScalarGraph<PAD, TT, B, INP_H, INP_W, H0, H1, W0, W1> g;
    static constexpr int OUT_H = INP_H + H0 + H1;
    static constexpr int OUT_W = INP_W + W0 + W1;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    Pad2DWindowGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "pad_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_pad_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_pad_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<INP_H*INP_W*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<OUT_H*OUT_W*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
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

// float32
Pad2DStreamGraphTest<Pad2DStreamScalar, float_t, B, INP_H, INP_W, H0, H1, W0, W1> pad2DScalar(
  "pad2DScalar", "pad_2d_fpin.txt", "pad_2d_fpout_shape2x34x34_Pad2DStreamScalar.txt");
Pad2DStreamGraphTest<Pad2DStreamFloat, float_t, B, INP_H, INP_W, H0, H1, W0, W1> pad2DFloat(
  "pad2DFloat", "pad_2d_fpin.txt", "pad_2d_fpout_shape2x34x34_Pad2DStreamFloat.txt");
Pad2DWindowGraphTest<Pad2DWindowScalar, float_t, B, INP_H, INP_W, H0, H1, W0, W1> pad2DWindow_float(
  "pad2DWindow_float", "pad_2d_fpin.txt", "pad_2d_fpout_shape2x34x34_Pad2DWindowScalar.txt");

// int8
Pad2DWindowGraphTest<Pad2DWindowScalar, int8_t, B, INP_H, INP_W, H0, H1, W0, W1> pad2DWindow_int8(
  "pad2DWindow_int8", "pad_2d_int8in.txt", "pad_2d_int8out_shape2x34x34_Pad2DWindowScalar.txt");
Pad2DStreamGraphTest<Pad2DStreamInt8, int8_t, B, INP_H, INP_W, H0, H1, W0, W1> pad2DStreamInt8(
  "pad2DStreamInt8", "pad_2d_int8in.txt", "pad_2d_int8out_shape2x34x34_Pad2DStreamInt8.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // float32
  adfCheck(pad2DScalar.init(), "init pad2DScalar");
  adfCheck(pad2DScalar.run(ITER_CNT), "run pad2DScalar");
	adfCheck(pad2DScalar.end(), "end pad2DScalar");
  
  adfCheck(pad2DFloat.init(), "init pad2DFloat");
  adfCheck(pad2DFloat.run(ITER_CNT), "run pad2DFloat");
	adfCheck(pad2DFloat.end(), "end pad2DFloat");

  adfCheck(pad2DWindow_float.init(), "init pad2DWindow_float");
  adfCheck(pad2DWindow_float.run(ITER_CNT), "run pad2DWindow_float");
	adfCheck(pad2DWindow_float.end(), "end pad2DWindow_float");

  // int8
  adfCheck(pad2DWindow_int8.init(), "init pad2DWindow_int8");
  adfCheck(pad2DWindow_int8.run(ITER_CNT), "run pad2DWindow_int8");
	adfCheck(pad2DWindow_int8.end(), "end pad2DWindow_int8");

  adfCheck(pad2DStreamInt8.init(), "init pad2DStreamInt8");
  adfCheck(pad2DStreamInt8.run(ITER_CNT), "run pad2DStreamInt8");
	adfCheck(pad2DStreamInt8.end(), "end pad2DStreamInt8");
  return 0;
}
#endif
