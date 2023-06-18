#include "graph_concat.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatGraphTest : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
    ConcatGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatGraphTest(
      const std::string& id,
      const std::string& OUT_TXT = "concat_out.txt",
      const std::vector<std::string> INP_TXT = std::vector<std::string>()
    ) {
      for (int i = 0; i < LCNT; i++) {
        std::string plin_name = "plin"+std::to_string(i)+"_concat_"+id+"_input"; 
        plin[i] = adf::input_plio::create(plin_name, PLIO64_ARG(INP_TXT[i]));
        adf::connect<adf::window<H*INP_W*TTSIZE>> (plin[i].out[0], g.pin[i]);
      }

      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatStreamGraphTest : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
    ConcatStreamGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatStreamGraphTest(
      const std::string& id,
      const std::string& OUT_TXT = "concat_out.txt",
      const std::vector<std::string> INP_TXT = std::vector<std::string>()
    ) {
      for (int i = 0; i < LCNT; i++) {
        std::string plin_name = "plin"+std::to_string(i)+"_concat_"+id+"_input"; 
        plin[i] = adf::input_plio::create(plin_name, PLIO64_ARG(INP_TXT[i]));
        adf::connect<adf::stream> (plin[i].out[0], g.pin[i]);
      }

      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatTwiceGraphTest : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
    ConcatTwiceGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatTwiceGraphTest(
      const std::string& id,
      const std::string& OUT_TXT = "concat_out.txt",
      const std::vector<std::string> INP_TXT = std::vector<std::string>()
    ) {
      for (int i = 0; i < LCNT; i++) {
        std::string plin_name = "plin"+std::to_string(i)+"_concat_"+id+"_input"; 
        plin[i] = adf::input_plio::create(plin_name, PLIO64_ARG(INP_TXT[i]));
        adf::connect<adf::window<H*INP_W*TTSIZE>> (plin[i].out[0], g.pin[i]);
      }
      
      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<H*OUT_W*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
const int H = 4;
const int INP_W = 32; // % 16 for int8, % 8 for float


// float32
std::vector<std::string> fp_txts {
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt" 
};
const int LCNT = 5; 
const int OUT_W = 144; //% 16 for int8, % 4 for float
ConcatGraphTest<ConcatScalar, float_t, LCNT, H, INP_W, OUT_W> concatScalar(
  "concatScalar", "concat_fpout_shape4x144_ConcatScalar.txt", fp_txts);
ConcatGraphTest<ConcatFloat, float_t, LCNT, H, INP_W, OUT_W> concatFloat(
  "concatFloat", "concat_fpout_shape4x144_ConcatFloat.txt", fp_txts);


// stream
ConcatStreamGraphTest<ConcatScalarStream, float_t, 2, H, INP_W, 48> concatStreamScalar2(
  "concatStreamScalar2", "concat_fpout_shape4x48_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatScalarStream, float_t, 3, H, INP_W, 80> concatStreamScalar3(
  "concatStreamScalar3", "concat_fpout_shape4x80_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatScalarStream, float_t, 4, H, INP_W, 112> concatStreamScalar4(
  "concatStreamScalar4", "concat_fpout_shape4x112_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatScalarStream, float_t, 5, H, INP_W, 144> concatStreamScalar5(
  "concatStreamScalar5", "concat_fpout_shape4x144_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatScalarStream, float_t, 6, H, INP_W, 176> concatStreamScalar6(
  "concatStreamScalar6", "concat_fpout_shape4x176_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatScalarStream, float_t, 7, H, INP_W, 208> concatStreamScalar7(
  "concatStreamScalar7", "concat_fpout_shape4x208_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatScalarStream, float_t, 8, H, INP_W, 240> concatStreamScalar8(
  "concatStreamScalar8", "concat_fpout_shape4x240_ConcatStreamScalar.txt", fp_txts);


// int8
std::vector<std::string> int8_txts {
  "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", 
  "concat_int8in.txt"
};
ConcatGraphTest<ConcatInt8, int8_t, LCNT, H, INP_W, OUT_W> concatInt8(
  "concatInt8", "concat_int8out_shape4x144_ConcatInt8.txt", int8_txts);

// multi concat
// const int MULTI_LCNT = 52; 
// const int MULTI_OUT_W = 816; //% 16 for int8, % 4 for float
// std::vector<std::string> multi_fp_txts {
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
//   "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt"
// };
// ConcatTwiceGraphTest<ConcatFloat, float_t, MULTI_LCNT, H, INP_W, MULTI_OUT_W> multiConcatFloat(
//   "multiConcatFloat", "concatmulti_fpout_shape4x816_ConcatFloat.txt", multi_fp_txts
// );


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // float32
  adfCheck(concatScalar.init(), "init concatScalar");
  adfCheck(concatScalar.run(ITER_CNT), "run concatScalar");
	adfCheck(concatScalar.end(), "end concatScalar");

  adfCheck(concatFloat.init(), "init concatFloat");
  adfCheck(concatFloat.run(ITER_CNT), "run concatFloat");
	adfCheck(concatFloat.end(), "end concatFloat");

  // stream
  adfCheck(concatStreamScalar2.init(), "init concatStreamScalar2");
  adfCheck(concatStreamScalar2.run(ITER_CNT), "run concatStreamScalar2");
	adfCheck(concatStreamScalar2.end(), "end concatStreamScalar2");
  
  adfCheck(concatStreamScalar3.init(), "init concatStreamScalar3");
  adfCheck(concatStreamScalar3.run(ITER_CNT), "run concatStreamScalar3");
	adfCheck(concatStreamScalar3.end(), "end concatStreamScalar3");

  adfCheck(concatStreamScalar4.init(), "init concatStreamScalar4");
  adfCheck(concatStreamScalar4.run(ITER_CNT), "run concatStreamScalar4");
	adfCheck(concatStreamScalar4.end(), "end concatStreamScalar4");
  
  adfCheck(concatStreamScalar5.init(), "init concatStreamScalar5");
  adfCheck(concatStreamScalar5.run(ITER_CNT), "run concatStreamScalar5");
	adfCheck(concatStreamScalar5.end(), "end concatStreamScalar5");

  adfCheck(concatStreamScalar6.init(), "init concatStreamScalar6");
  adfCheck(concatStreamScalar6.run(ITER_CNT), "run concatStreamScalar6");
	adfCheck(concatStreamScalar6.end(), "end concatStreamScalar6");

  adfCheck(concatStreamScalar7.init(), "init concatStreamScalar7");
  adfCheck(concatStreamScalar7.run(ITER_CNT), "run concatStreamScalar7");
	adfCheck(concatStreamScalar7.end(), "end concatStreamScalar7");

  adfCheck(concatStreamScalar8.init(), "init concatStreamScalar8");
  adfCheck(concatStreamScalar8.run(ITER_CNT), "run concatStreamScalar8");
	adfCheck(concatStreamScalar8.end(), "end concatStreamScalar8");

  // int8
  adfCheck(concatInt8.init(), "init concatInt8");
  adfCheck(concatInt8.run(ITER_CNT), "run concatInt8");
	adfCheck(concatInt8.end(), "end concatInt8");

  // not very feasible mapping into array
  // adfCheck(multiConcatFloat.init(), "init multiConcatFloat");
  // adfCheck(multiConcatFloat.run(ITER_CNT), "run multiConcatFloat");
	// adfCheck(multiConcatFloat.end(), "end multiConcatFloat");
  return 0;
}
#endif
