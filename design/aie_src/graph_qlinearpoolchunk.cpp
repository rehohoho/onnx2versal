#include "graph_qlinearpool.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int> class QLINEARPOOL,
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK, 
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class QLinearPoolChunkCStreamGraphTest : public adf::graph {

  private:
    QLinearPoolChunkCStreamGraph<SPLIT, QLINEARPOOL, CONCAT, CCHUNK,
                                 TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QLinearPoolChunkCStreamGraphTest(
      const std::string& id,
      float in_scale,
      float out_scale,
      int8_t in_zero,
      int8_t out_zero,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ): g(in_scale, out_scale, in_zero, out_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearpool"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearpool"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*C*INP_H*INP_W*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
QLinearPoolChunkCStreamGraphTest<SplitInt8, QLinearAvgpoolScalarBCHW, ConcatInt8Stream, 
                                 16, int8_t,8,16,1,1,1,64,8,8> k14qlinearpool(
  "k14qlinearpool", 0.12030204, 0.022007886, -128, -128,
  "k14qlinearpool_in_shape1x64x8x16.txt", 
  "k14qlinearpool_goldenout_shape1x64x1x1.txt");

QLinearPoolChunkCStreamGraphTest<SplitInt8,QLinearAvgpoolScalarBCHW,ConcatInt8Stream,
                                 32,int8_t,25,16,1,1,1,64,25,5> k11qlinearpool(
  "k11qlinearpool", 0.09263198, 0.016548144, -128, -128,
  "k11qlinearpool_in_shape1x64x25x16.txt",
  "k11qlinearpool_goldenout_shape1x64x1x1.txt"
);


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(k14qlinearpool.init(), "init k14qlinearpool");
  adfCheck(k14qlinearpool.run(ITER_CNT), "run k14qlinearpool");
	adfCheck(k14qlinearpool.end(), "end k14qlinearpool");

  adfCheck(k11qlinearpool.init(), "init k11qlinearpool");
  adfCheck(k11qlinearpool.run(ITER_CNT), "run k11qlinearpool");
	adfCheck(k11qlinearpool.end(), "end k11qlinearpool");
  return 0;
}
#endif