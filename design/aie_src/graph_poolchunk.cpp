#include "graph_pool.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int, int, int> class POOL,
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK,
  typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_H, int OUT_W, int B, int C, int KH, int KW, int STEP_H, int STEP_W,
  template<typename, int, int, int, int, int, int, int, int> class PAD, 
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class PoolChunkCGraphTest : public adf::graph {

  private:
    PoolChunkCGraph<SPLIT, POOL, CONCAT, CCHUNK, TT, INP_H, INP_W, INP_W_PAD, OUT_H, OUT_W, B, C, KH, KW, STEP_H, STEP_W,
                    PAD, H0, H1, W0, W1> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    PoolChunkCGraphTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ): g(1) { 
      plin[0] = adf::input_plio::create("plin0_maxpool"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_maxpool"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*C*INP_H*INP_W_PAD*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*C*OUT_H*OUT_W*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
const int CCHUNK = 1;
const int INP_H = 24;
const int INP_W = 24;
const int INP_W_PAD4 = (INP_W+3)/4*4;
const int INP_W_PAD16 = (INP_W+15)/16*16;
const int OUT_H = 12;
const int OUT_W = 12;
const int OUT_W_PAD4 = (OUT_W+3)/4*4;
const int OUT_W_PAD16 = (OUT_W+15)/16*16;
const int B = 1;
const int C = 6;
const int KH = 2;
const int KW = 2;
const int STEP_H = 2;
const int STEP_W = 2;

// BCHW
PoolChunkCGraphTest<SplitFilterFloatPktStream, MaxpoolScalarBCHW, ConcatFloatStream, 
                    CCHUNK, float, INP_H, INP_W, INP_W_PAD4, OUT_H, OUT_W_PAD4, B, C, KH, KW, STEP_H, STEP_W,
                    Pad2DStreamFloat, 0, 0, 0, 0> maxpoolScalarBCHW(
  "maxpoolScalarBCHW", "pool_fpin.txt", "poolBCHW_max_fpout_shape1x6x12x12_MaxpoolScalarBCHW.txt");
PoolChunkCGraphTest<SplitFilterFloatPktStream, Maxpool2x2FloatBCHW, ConcatFloatStream, 
                    CCHUNK, float, INP_H, INP_W, INP_W_PAD4, OUT_H, OUT_W_PAD4, B, C, KH, KW, STEP_H, STEP_W,
                    Pad2DStreamFloat, 0, 0, 0, 0> maxpool2x2BCHW(
  "maxpool2x2BCHW", "pool_fpin.txt", "poolBCHW_max_fpout_shape1x6x12x12_Maxpool2x2FloatBCHW.txt");
PoolChunkCGraphTest<SplitFilterFloatPktStream, AvgpoolScalarBCHW, ConcatFloatStream, 
                    CCHUNK, float, INP_H, INP_W, INP_W_PAD4, OUT_H, OUT_W_PAD4, B, C, KH, KW, STEP_H, STEP_W,
                    Pad2DStreamFloat, 0, 0, 0, 0> avgpoolScalarBCHW(
  "avgpoolScalarBCHW", "pool_fpin.txt", "poolBCHW_avg_fpout_shape1x6x12x12_AvgpoolScalarBCHW.txt");
PoolChunkCGraphTest<SplitFilterInt8PktStream, Maxpool2x2Int8BCHW, ConcatInt8Stream, 
                    CCHUNK, int8_t, INP_H, INP_W, INP_W_PAD16, OUT_H, OUT_W_PAD16, B, C, KH, KW, STEP_H, STEP_W,
                    Pad2DStreamInt8, 0, 0, 0, 8> maxpool2x2int8BCHW(
  "maxpool2x2int8BCHW", "pool_int8in_pad.txt", "poolBCHW_max_int8out_shape1x6x12x12_Maxpool2x2Int8BCHW.txt");

/*
Maxpool3x3Int8BCHW<114,128,56,64,1,1,3,3,2,2> start = 9323,end = 31000,total = 21677
Maxpool3x3Int8BCHW<114,128,56,64,1,1,3,3,2,2> start = 17157,end = 38834,total = 21677
ConcatInt8Stream<h,1,3584,3584,7168> start = 1423,end = 39865,total = 38442
Maxpool3x3Int8BCHW<114,128,56,64,1,1,3,3,2,2> start = 24990,end = 46667,total = 21677
Maxpool3x3Int8BCHW<114,128,56,64,1,1,3,3,2,2> start = 32823,end = 54500,total = 21677
ConcatInt8Stream<h,1,3584,3584,7168> start = 1423,end = 55532,total = 54109
ConcatInt8Stream<h,1,7168,7168,14336> start = 1421,end = 55543,total = 54122
*/
// PoolChunkCGraphTest<SplitFilterInt8PktStream,Maxpool3x3Int8BCHW,ConcatInt8Stream,
//                     1,uint8_t,112,112,56,64,1,8,3,3,2,2,
//                     Pad2DStreamInt8,1,1,1,15> k002pool(
//   "k002pool",
//   "k002pool_in_shape1x64x112x112.txt",
//   "k002pool_goldenout_shape1x64x56x56.txt"
// );


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // adfCheck(k002pool.init(), "init k002pool");
  // adfCheck(k002pool.run(ITER_CNT), "run k002pool");
	// adfCheck(k002pool.end(), "end k002pool");

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
  return 0;
}
#endif