#include "graph_add.h"
#include "graph_utils.h"


template <template<typename, int, int> class ADD, 
  typename TT, int W, int IS_RELU>
class AddGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    AddGraph<ADD, TT, W, IS_RELU> g;

  public:
    adf::input_plio plin[2];
    adf::output_plio plout[1];

    AddGraphTest(
      const std::string& id,
      const std::string& INPA_TXT, 
      const std::string& INPB_TXT, 
      const std::string& OUT_TXT
    ) { 
      plin[0] = adf::input_plio::create("plin0_add"+id, PLIO64_ARG(INPA_TXT));
      plin[1] = adf::input_plio::create("plin1_add"+id, PLIO64_ARG(INPB_TXT));
      plout[0] = adf::output_plio::create("plout0_add"+id, PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (plin[1].out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <template<typename, int, int> class ADD, 
  typename TT, int W, int IS_RELU>
class AddGraphGmioTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    AddGraph<ADD, TT, W, IS_RELU> g;

  public:
    adf::input_plio plin[1];
    adf::input_gmio gmin[1];
    adf::output_gmio gmout[1];

    AddGraphGmioTest(
      const std::string& id,
      const std::string& INPA_TXT, 
      const int repeat
    ): g(repeat) { 
      plin[0] = adf::input_plio::create("plin0_add"+id, PLIO64_ARG(INPA_TXT));
      gmin[0] = adf::input_gmio::create("gmin0_add"+id+"_w", 64, 1000);
      gmout[0] = adf::output_gmio::create("gmout0_add"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmin[0].out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], gmout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
// AddGraphTest<AddScalar, float_t, 16384, 1> addScalar(
//   "addScalar", 
//   "k6add_inA_shape1x16x32x32.txt", 
//   "k6add_inB_shape1x16x32x32.txt",
//   "k6add_goldenout_shape1x16x32x32_scalar.txt");

// AddGraphTest<AddFloat, float_t, 16384, 1> addFloat(
//   "addFloat", 
//   "k6add_inA_shape1x16x32x32.txt", 
//   "k6add_inB_shape1x16x32x32.txt",
//   "k6add_goldenout_shape1x16x32x32_float.txt");

AddGraphGmioTest<AddFloat, float_t, 64, 1> addFloatGmio(
  "addFloatGmio", "concat_fpin.txt", 1);


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	// adfCheck(addScalar.init(), "init addScalar");
  // adfCheck(addScalar.run(ITER_CNT), "run addScalar");
	// adfCheck(addScalar.end(), "end addScalar");

  // adfCheck(addFloat.init(), "init addFloat");
  // adfCheck(addFloat.run(ITER_CNT), "run addFloat");
	// adfCheck(addFloat.end(), "end addFloat");
  
  float* addbuff = (float *) adf::GMIO::malloc(64*sizeof(float));
  memset(addbuff, 0, 64*sizeof(float));

  adfCheck(addFloatGmio.init(), "init addFloatGmio");
  adfCheck(addFloatGmio.run(ITER_CNT), "run addFloatGmio");
  for (int i = 0; i < ITER_CNT; i++) {
    addFloatGmio.gmin[0].gm2aie_nb(addbuff, 64*sizeof(float));
    addFloatGmio.gmout[0].aie2gm(addbuff, 64*sizeof(float));
  }
  // addFloatGmio.gmout[0].wait();
	adfCheck(addFloatGmio.end(), "end addFloatGmio");

  for (int i = 0; i < 64; i++) {
    printf("%f ", addbuff[i]);
  }
  
  adf::GMIO::free(addbuff);
  return 0;
}
#endif
