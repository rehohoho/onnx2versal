#include "graph_pool.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int, int, int, int, int, int, int> class POOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W,
  template<typename, int, int, int, int, int, int, int, int> class PAD, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class PoolGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    PoolGraph<POOL, TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W,
              PAD, H0, H1, W0, W1> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    PoolGraphTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ) { 
      plin[0] = adf::input_plio::create("plin0_maxpool"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_maxpool"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_H*INP_W*C*TTSIZE>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
const int INP_H = 24;
const int INP_W = 24;
const int INP_W_PAD16 = (INP_W+15)/16*16;
const int OUT_H = 12;
const int OUT_W = 12;
const int OUT_W_PAD16 = (OUT_W+15)/16*16;
const int B = 1;
const int C = 6;
const int KH = 2;
const int KW = 2;
const int STEP_H = 2;
const int STEP_W = 2;

// BCHW
PoolGraphTest<MaxpoolScalarBCHW, float, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W, Pad2DStreamInt8,0,0,0> maxpoolScalarBCHW(
  "maxpoolScalarBCHW", "pool_fpin.txt", "poolBCHW_max_fpout_shape1x6x12x12_MaxpoolScalarBCHW.txt");
PoolGraphTest<Maxpool2x2FloatBCHW, float, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W, Pad2DStreamInt8,0,0,0> maxpool2x2BCHW(
  "maxpool2x2BCHW", "pool_fpin.txt", "poolBCHW_max_fpout_shape1x6x12x12_Maxpool2x2FloatBCHW.txt");
PoolGraphTest<AvgpoolScalarBCHW, float, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W> avgpoolScalarBCHW(
  "avgpoolScalarBCHW", "pool_fpin.txt", "poolBCHW_avg_fpout_shape1x6x12x12_AvgpoolScalarBCHW.txt");

PoolGraphTest<Maxpool2x2Int8BCHW, int8_t, INP_H, INP_W_PAD16, OUT_H, OUT_W_PAD16, B, C, KH, KW, STEP_H, STEP_W> maxpool2x2int8BCHW(
  "maxpool2x2int8BCHW", "pool_int8in_pad.txt", "poolBCHW_max_int8out_shape1x6x12x12_Maxpool2x2Int8BCHW.txt");

// BHWC
PoolGraphTest<MaxpoolScalarBHWC, float, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W> maxpoolScalarBHWC(
  "maxpoolScalarBHWC", "pool_fpin.txt", "poolBHWC_max_fpout_shape1x12x12x6_MaxpoolScalarBHWC.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // BCHW
  adfCheck(maxpoolScalarBCHW.init(), "init maxpoolScalarBCHW");
  adfCheck(maxpoolScalarBCHW.run(ITER_CNT), "run maxpoolScalarBCHW");
	adfCheck(maxpoolScalarBCHW.end(), "end maxpoolScalarBCHW");

  adfCheck(maxpool2x2BCHW.init(), "init maxpool2x2BCHW");
  adfCheck(maxpool2x2BCHW.run(ITER_CNT), "run maxpool2x2BCHW");
	adfCheck(maxpool2x2BCHW.end(), "end maxpool2x2BCHW");

  adfCheck(avgpoolScalarBCHW.init(), "init avgpoolScalarBCHW");
  adfCheck(avgpoolScalarBCHW.run(ITER_CNT), "run avgpoolScalarBCHW");
	adfCheck(avgpoolScalarBCHW.end(), "end avgpoolScalarBCHW");

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