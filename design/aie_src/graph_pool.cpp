#include "graph_pool.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int, int> class POOL,
  typename TT, int INP_H, int INP_W, int OUT_W, int B, int C>
class MaxpoolGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    static constexpr int K = INP_H/OUT_W;
    MaxpoolGraph<POOL, TT, INP_H, INP_W, OUT_W, B, C> g;

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
      adf::connect<adf::window<B*INP_H*INP_W*C*TTSIZE>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*INP_H/K*OUT_W*C*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
// BCHW
MaxpoolGraphTest<MaxpoolScalarBCHW, float, 24, 24, 12, 1, 6> maxpoolScalarBCHW(
  "maxpoolScalarBCHW", "pool_fpin.txt", "pool_fpout_MaxpoolScalarBCHW_shape1x6x12x12.txt");
MaxpoolGraphTest<Maxpool2x2FloatBCHW, float, 24, 24, 12, 1, 6> maxpool2x2BCHW(
  "maxpool2x2BCHW", "pool_fpin.txt", "pool_fpout_Maxpool2x2FloatBCHW_shape1x6x12x12.txt");

MaxpoolGraphTest<Maxpool2x2Int8BCHW, int8_t, 24, 32, 16, 1, 6> maxpool2x2int8BCHW(
  "maxpool2x2int8BCHW", "pool_int8in_pad.txt", "pool_int8out_Maxpool2x2Int8BCHW_shape1x6x12x12.txt");

// BHWC
MaxpoolGraphTest<MaxpoolScalarBHWC, float, 24, 24, 12, 1, 6> maxpoolScalarBHWC(
  "maxpoolScalarBHWC", "pool_fpin.txt", "pool_fpout_MaxpoolScalarBHWC_shape1x6x12x12.txt");

#ifdef __X86SIM__
int main(int argc, char ** argv) {
  // BCHW
  adfCheck(maxpoolScalarBCHW.init(), "init maxpoolScalarBCHW");
  adfCheck(maxpoolScalarBCHW.run(ITER_CNT), "run maxpoolScalarBCHW");
	adfCheck(maxpoolScalarBCHW.end(), "end maxpoolScalarBCHW");

  adfCheck(maxpool2x2BCHW.init(), "init maxpool2x2BCHW");
  adfCheck(maxpool2x2BCHW.run(ITER_CNT), "run maxpool2x2BCHW");
	adfCheck(maxpool2x2BCHW.end(), "end maxpool2x2BCHW");

  adfCheck(maxpool2x2int8BCHW.init(), "init maxpool2x2int8BCHW");
  adfCheck(maxpool2x2int8BCHW.run(ITER_CNT), "run maxpool2x2int8BCHW");
	adfCheck(maxpool2x2int8BCHW.end(), "end maxpool2x2int8BCHW");

  // BHWC
  adfCheck(maxpoolScalarBHWC.init(), "init maxpoolScalarBHWC");
  adfCheck(maxpoolScalarBHWC.run(ITER_CNT), "run maxpoolScalarBHWC");
	adfCheck(maxpoolScalarBHWC.end(), "end maxpoolScalarBHWC");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  // BCHW
  adfCheck(maxpoolScalarBCHW.init(), "init maxpoolScalarBCHW");
  get_graph_throughput_by_port(maxpoolScalarBCHW, "plout[0]", maxpoolScalarBCHW.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpoolScalarBCHW.end(), "end maxpoolScalarBCHW");

  adfCheck(maxpool2x2BCHW.init(), "init maxpool2x2BCHW");
  get_graph_throughput_by_port(maxpool2x2BCHW, "plout[0]", maxpool2x2BCHW.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpool2x2BCHW.end(), "end maxpool2x2BCHW");

  adfCheck(maxpool2x2int8BCHW.init(), "init maxpool2x2int8BCHW");
  get_graph_throughput_by_port(maxpool2x2int8BCHW, "plout[0]", maxpool2x2int8BCHW.plout[0], 1*6*12*12, sizeof(float_t), ITER_CNT);
	adfCheck(maxpool2x2int8BCHW.end(), "end maxpool2x2int8BCHW");

  // BHWC
	adfCheck(maxpoolScalarBHWC.init(), "init maxpoolScalarBHWC");
  get_graph_throughput_by_port(maxpoolScalarBHWC, "plout[0]", maxpoolScalarBHWC.plout[0], 1*12*12*6, sizeof(float_t), ITER_CNT);
	adfCheck(maxpoolScalarBHWC.end(), "end maxpoolScalarBHWC");
  return 0;
}
#endif
