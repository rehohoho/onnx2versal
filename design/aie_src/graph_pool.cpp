#include "graph_pool.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class POOL,
  typename TT, int INP_W, int OUT_W, int B, int C>
class MaxpoolGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    MaxpoolGraph<POOL, TT, INP_W, OUT_W, B, C> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    MaxpoolGraphTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ) { 
      plin[0] = adf::input_plio::create("plin0_maxpool"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_maxpool"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_W*INP_W*C*TTSIZE>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*C*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
// BHWC
MaxpoolGraphTest<MaxpoolScalarBHWC, float, 24, 12, 1, 6> maxpoolScalarBHWC(
  "maxpoolScalarBHWC", "pool_fpin.txt", "pool_fpout_MaxpoolScalarBHWC.txt");
MaxpoolGraphTest<MaxpoolScalarBHWC, float, 24, 12, 1, 6> maxpoolScalarBHWC_rand(
  "maxpoolScalarBHWC_rand", "pool_fpin_rand.txt", "pool_fpout_MaxpoolScalarBHWC_rand.txt");

// BCHW
MaxpoolGraphTest<MaxpoolScalarBCHW, float, 24, 12, 1, 6> maxpoolScalarBCHW(
  "maxpoolScalarBCHW", "pool_fpin.txt", "pool_fpout_MaxpoolScalarBCHW.txt");
MaxpoolGraphTest<MaxpoolScalarBCHW, float, 24, 12, 1, 6> maxpoolScalarBCHW_rand(
  "maxpoolScalarBCHW_rand", "pool_fpin_rand.txt", "pool_fpout_MaxpoolScalarBCHW_rand.txt");

MaxpoolGraphTest<Maxpool2x2BCHW, float, 24, 12, 1, 6> maxpool2x2BCHW(
  "maxpool2x2BCHW", "pool_fpin.txt", "pool_fpout_Maxpool2x2BCHW.txt");
MaxpoolGraphTest<Maxpool2x2BCHW, float, 24, 12, 1, 6> maxpool2x2BCHW_rand(
  "maxpool2x2BCHW_rand", "pool_fpin_rand.txt", "pool_fpout_Maxpool2x2BCHW_rand.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(maxpoolScalarBHWC.init(), "init maxpoolScalarBHWC");
  adfCheck(maxpoolScalarBHWC.run(ITER_CNT), "run maxpoolScalarBHWC");
	adfCheck(maxpoolScalarBHWC.end(), "end maxpoolScalarBHWC");

  adfCheck(maxpoolScalarBHWC_rand.init(), "init maxpoolScalarBHWC_rand");
  adfCheck(maxpoolScalarBHWC_rand.run(ITER_CNT), "run maxpoolScalarBHWC_rand");
	adfCheck(maxpoolScalarBHWC_rand.end(), "end maxpoolScalarBHWC_rand");

  adfCheck(maxpoolScalarBCHW.init(), "init maxpoolScalarBCHW");
  adfCheck(maxpoolScalarBCHW.run(ITER_CNT), "run maxpoolScalarBCHW");
	adfCheck(maxpoolScalarBCHW.end(), "end maxpoolScalarBCHW");

  adfCheck(maxpoolScalarBCHW_rand.init(), "init maxpoolScalarBCHW_rand");
  adfCheck(maxpoolScalarBCHW_rand.run(ITER_CNT), "run maxpoolScalarBCHW_rand");
	adfCheck(maxpoolScalarBCHW_rand.end(), "end maxpoolScalarBCHW_rand");

  adfCheck(maxpool2x2BCHW.init(), "init maxpool2x2BCHW");
  adfCheck(maxpool2x2BCHW.run(ITER_CNT), "run maxpool2x2BCHW");
	adfCheck(maxpool2x2BCHW.end(), "end maxpool2x2BCHW");

  adfCheck(maxpool2x2BCHW_rand.init(), "init maxpool2x2BCHW_rand");
  adfCheck(maxpool2x2BCHW_rand.run(ITER_CNT), "run maxpool2x2BCHW_rand");
	adfCheck(maxpool2x2BCHW_rand.end(), "end maxpool2x2BCHW_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(maxpoolScalarBHWC.init(), "init maxpoolScalarBHWC");
  get_graph_throughput_by_port(maxpoolScalarBHWC, "plout[0]", maxpoolScalarBHWC.plout[0], 1*12*12*6, sizeof(float_t), ITER_CNT);
	adfCheck(maxpoolScalarBHWC.end(), "end maxpoolScalarBHWC");

  adfCheck(maxpoolScalarBHWC_rand.init(), "init maxpoolScalarBHWC_rand");
  get_graph_throughput_by_port(maxpoolScalarBHWC_rand, "plout[0]", maxpoolScalarBHWC_rand.plout[0], 1*12*12*6, sizeof(float_t), ITER_CNT);
	adfCheck(maxpoolScalarBHWC_rand.end(), "end maxpoolScalarBHWC_rand");

  adfCheck(maxpoolScalarBCHW.init(), "init maxpoolScalarBCHW");
  get_graph_throughput_by_port(maxpoolScalarBCHW, "plout[0]", maxpoolScalarBCHW.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpoolScalarBCHW.end(), "end maxpoolScalarBCHW");

  adfCheck(maxpoolScalarBCHW_rand.init(), "init maxpoolScalarBCHW_rand");
  get_graph_throughput_by_port(maxpoolScalarBCHW_rand, "plout[0]", maxpoolScalarBCHW_rand.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpoolScalarBCHW_rand.end(), "end maxpoolScalarBCHW_rand");

  adfCheck(maxpool2x2BCHW.init(), "init maxpool2x2BCHW");
  get_graph_throughput_by_port(maxpool2x2BCHW, "plout[0]", maxpool2x2BCHW.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpool2x2BCHW.end(), "end maxpool2x2BCHW");

  adfCheck(maxpool2x2BCHW_rand.init(), "init maxpool2x2BCHW_rand");
  get_graph_throughput_by_port(maxpool2x2BCHW_rand, "plout[0]", maxpool2x2BCHW_rand.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpool2x2BCHW_rand.end(), "end maxpool2x2BCHW_rand");
  return 0;
}
#endif
