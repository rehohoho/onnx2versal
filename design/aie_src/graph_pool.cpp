#include "graph_pool.h"
#include "graph_utils.h"

#define ITER_CNT 1


template <template<int, int, int, int> class POOL,
  int INP_W, int OUT_W, int B, int C>
class MaxpoolGraphTest : public adf::graph {

  private:
    MaxpoolGraph<POOL, INP_W, OUT_W, B, C> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    MaxpoolGraphTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ) { 
      g.construct();
      plin[0] = adf::input_plio::create("plin0_maxpool"+id+"_input", adf::plio_64_bits, TXT_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_maxpool"+id+"_output", adf::plio_64_bits, TXT_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*C*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
MaxpoolGraphTest<MaxpoolScalarBHWC, 24, 12, 1, 6> fpscalar_bhwc(
  "fpscalar_bhwc", "pool_fpin.txt", "pool_fpout_MaxpoolScalarBHWC.txt");
MaxpoolGraphTest<MaxpoolScalarBHWC, 24, 12, 1, 6> fpscalar_bhwc_rand(
  "fpscalar_bhwc_rand", "pool_fpin_rand.txt", "pool_fpout_MaxpoolScalarBHWC_rand.txt");

MaxpoolGraphTest<MaxpoolScalarBCHW, 24, 12, 1, 6> fpscalar_bchw(
  "fpscalar_bchw", "pool_fpin.txt", "pool_fpout_MaxpoolScalarBCHW.txt");
MaxpoolGraphTest<MaxpoolScalarBCHW, 24, 12, 1, 6> fpscalar_bchw_rand(
  "fpscalar_bchw_rand", "pool_fpin_rand.txt", "pool_fpout_MaxpoolScalarBCHW_rand.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(fpscalar_bhwc.init(), "init fpscalar_bhwc");
  adfCheck(fpscalar_bhwc.run(1), "run fpscalar_bhwc");
	adfCheck(fpscalar_bhwc.end(), "end fpscalar_bhwc");

  adfCheck(fpscalar_bhwc_rand.init(), "init fpscalar_bhwc_rand");
  adfCheck(fpscalar_bhwc_rand.run(1), "run fpscalar_bhwc_rand");
	adfCheck(fpscalar_bhwc_rand.end(), "end fpscalar_bhwc_rand");

  adfCheck(fpscalar_bchw.init(), "init fpscalar_bchw");
  adfCheck(fpscalar_bchw.run(1), "run fpscalar_bchw");
	adfCheck(fpscalar_bchw.end(), "end fpscalar_bchw");

  adfCheck(fpscalar_bchw_rand.init(), "init fpscalar_bchw_rand");
  adfCheck(fpscalar_bchw_rand.run(1), "run fpscalar_bchw_rand");
	adfCheck(fpscalar_bchw_rand.end(), "end fpscalar_bchw_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(fpscalar_bhwc.init(), "init fpscalar_bhwc");
  get_graph_throughput_by_port(fpscalar_bhwc, "plout[0]", fpscalar_bhwc.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(fpscalar_bhwc.end(), "end fpscalar_bhwc");

  adfCheck(fpscalar_bhwc_rand.init(), "init fpscalar_bhwc_rand");
  get_graph_throughput_by_port(fpscalar_bhwc_rand, "plout[0]", fpscalar_bhwc_rand.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(fpscalar_bhwc_rand.end(), "end fpscalar_bhwc_rand");

  adfCheck(fpscalar_bchw.init(), "init fpscalar_bchw");
  get_graph_throughput_by_port(fpscalar_bchw, "plout[0]", fpscalar_bchw.plout[0], 1*6*12*12, sizeof(float32), ITER_CNT);
	adfCheck(fpscalar_bchw.end(), "end fpscalar_bchw");

  adfCheck(fpscalar_bchw_rand.init(), "init fpscalar_bchw_rand");
  get_graph_throughput_by_port(fpscalar_bchw_rand, "plout[0]", fpscalar_bchw_rand.plout[0], 1*6*12*12, sizeof(float32), ITER_CNT);
	adfCheck(fpscalar_bchw_rand.end(), "end fpscalar_bchw_rand");
  return 0;
}
#endif
