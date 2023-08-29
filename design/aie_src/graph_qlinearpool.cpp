#include "graph_qlinearpool.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int, int, int, int, int> class QLINEARPOOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class QLinearPoolGraphTest : public adf::graph {

  private:
    QLinearPoolGraph<QLINEARPOOL, TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QLinearPoolGraphTest(
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


template <template<typename, int, int, int, int, int, int, int, int> class QLINEARPOOL,
  typename TT, int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int KH, int KW>
class QLinearPoolStreamGraphTest : public adf::graph {

  private:
    QLinearPoolStreamGraph<QLINEARPOOL, TT, INP_H, INP_W, OUT_H, OUT_W, B, C, KH, KW> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QLinearPoolStreamGraphTest(
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
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
QLinearPoolGraphTest<QLinearAvgpoolScalarBCHW,int8_t,8,16,1,1,1,64,8,8> k14qlinearpool(
  "k14qlinearpool", 0.12030204, 0.022007886, -128, -128,
  "k14qlinearpool_in_shape1x64x8x16.txt", 
  "k14qlinearpool_goldenout_shape1x64x1x1.txt");

float_t k071qlinearpool_inscale = 0.3566944;
float_t k071qlinearpool_outscale = 0.0700892;
uint8_t k071qlinearpool_inzero = 0;
uint8_t k071qlinearpool_outzero = 0;
QLinearPoolStreamGraphTest<QLinearGlobalAvgpoolScalarBCHW,uint8_t,7,16,1,1,1,2048,7,7> k071qlinearpool(
  "k071qlinearpool",
  k071qlinearpool_inscale, k071qlinearpool_outscale, k071qlinearpool_inzero, k071qlinearpool_outzero,
  "k071qlinearpool_in_shape1x2048x7x16.txt",
  "k071qlinearpool_goldenout_shape1x2048x1x1.txt"
);


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(k14qlinearpool.init(), "init k14qlinearpool");
  adfCheck(k14qlinearpool.run(ITER_CNT), "run k14qlinearpool");
	adfCheck(k14qlinearpool.end(), "end k14qlinearpool");

  adfCheck(k071qlinearpool.init(), "init k071qlinearpool");
  adfCheck(k071qlinearpool.run(ITER_CNT), "run k071qlinearpool");
	adfCheck(k071qlinearpool.end(), "end k071qlinearpool");
  return 0;
}
#endif