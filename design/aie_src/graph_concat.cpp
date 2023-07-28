#include "graph_concat.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatGraphTest : public adf::graph {

  private:
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
        adf::connect<adf::window<H*INP_W*sizeof(TT)>> (plin[i].out[0], g.pin[i]);
      }

      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatStreamGraphTest : public adf::graph {

  private:
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
class ConcatTwoStreamGraphTest : public adf::graph {

  private:
    ConcatTwoStreamGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;
  
  public:
    adf::input_plio plin[2];
    adf::output_plio plout[1];

    ConcatTwoStreamGraphTest(
      const std::string& id,
      const std::string& INP_TXT0,
      const std::string& INP_TXT1,
      const std::string& OUT_TXT = "concat_out.txt"
    ) {
      plin[0] = adf::input_plio::create("plin0_concat_"+id+"_input", PLIO64_ARG(INP_TXT0));
      plin[1] = adf::input_plio::create("plin1_concat_"+id+"_input", PLIO64_ARG(INP_TXT1));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (plin[1].out[0], g.pin[1]);

      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatTwiceGraphTest : public adf::graph {

  private:
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
        adf::connect<adf::window<H*INP_W*sizeof(TT)>> (plin[i].out[0], g.pin[i]);
      }
      
      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<H*OUT_W*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatStreamSequentiallyGraphTest : public adf::graph {

  private:
    ConcatStreamSequentiallyGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatStreamSequentiallyGraphTest(
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


// instance to be compiled and used in host within xclbin
const int H = 4;
const int INP_W = 32; // % 16 for int8, % 8 for float


// float32 window
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


// float32 stream
ConcatStreamSequentiallyGraphTest<ConcatFloatStreamSequentially, float_t, 8, H, INP_W, 240> concatStreamSequentially(
  "concatStreamSequentially", "concat_fpout_shape4x240_ConcatFloatStream.txt", fp_txts);

ConcatStreamGraphTest<ConcatFloatStream, float_t, 2, H, INP_W, 48> concatStreamScalar2(
  "concatStreamScalar2", "concat_fpout_shape4x48_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatFloatStream, float_t, 3, H, INP_W, 80> concatStreamScalar3(
  "concatStreamScalar3", "concat_fpout_shape4x80_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatFloatStream, float_t, 4, H, INP_W, 112> concatStreamScalar4(
  "concatStreamScalar4", "concat_fpout_shape4x112_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatFloatStream, float_t, 5, H, INP_W, 144> concatStreamScalar5(
  "concatStreamScalar5", "concat_fpout_shape4x144_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatFloatStream, float_t, 6, H, INP_W, 176> concatStreamScalar6(
  "concatStreamScalar6", "concat_fpout_shape4x176_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatFloatStream, float_t, 7, H, INP_W, 208> concatStreamScalar7(
  "concatStreamScalar7", "concat_fpout_shape4x208_ConcatStreamScalar.txt", fp_txts);
ConcatStreamGraphTest<ConcatFloatStream, float_t, 8, H, INP_W, 240> concatStreamScalar8(
  "concatStreamScalar8", "concat_fpout_shape4x240_ConcatStreamScalar.txt", fp_txts);

ConcatTwoStreamGraphTest<ConcatTwo32bitStreams, float_t, 2, H, INP_W/8, 64> concatTwo32bitStreams(
  "concatTwo32bitStreams", "concat_fpin.txt", "concat_fpin.txt", "concat_fpout_2stream_shape4x64_ConcatTwo32bitStreams.txt");

// int8 window
std::vector<std::string> int8_txts {
  "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", 
  "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt"
};
ConcatGraphTest<ConcatInt8, int8_t, LCNT, H, INP_W, OUT_W> concatInt8(
  "concatInt8", "concat_int8out_shape4x144_ConcatInt8.txt", int8_txts);

// int8 stream
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 2, H, INP_W, 48> concatInt8Stream2(
  "concatInt8Stream2", "concat_int8out_shape4x48_concatInt8Stream.txt", int8_txts);
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 3, H, INP_W, 80> concatInt8Stream3(
  "concatInt8Stream3", "concat_int8out_shape4x80_concatInt8Stream.txt", int8_txts);
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 4, H, INP_W, 112> concatInt8Stream4(
  "concatInt8Stream4", "concat_int8out_shape4x112_concatInt8Stream.txt", int8_txts);
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 5, H, INP_W, 144> concatInt8Stream5(
  "concatInt8Stream5", "concat_int8out_shape4x144_concatInt8Stream.txt", int8_txts);
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 6, H, INP_W, 176> concatInt8Stream6(
  "concatInt8Stream6", "concat_int8out_shape4x176_concatInt8Stream.txt", int8_txts);
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 7, H, INP_W, 208> concatInt8Stream7(
  "concatInt8Stream7", "concat_int8out_shape4x208_concatInt8Stream.txt", int8_txts);
ConcatStreamGraphTest<ConcatInt8Stream, int8_t, 8, H, INP_W, 240> concatInt8Stream8(
  "concatInt8Stream8", "concat_int8out_shape4x240_concatInt8Stream.txt", int8_txts);

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
  adfCheck(concatStreamSequentially.init(), "init concatStreamSequentially");
  adfCheck(concatStreamSequentially.run(ITER_CNT), "run concatStreamSequentially");
	adfCheck(concatStreamSequentially.end(), "end concatStreamSequentially");
  
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

  // 2 stream
  adfCheck(concatTwo32bitStreams.init(), "init concatTwo32bitStreams");
  adfCheck(concatTwo32bitStreams.run(ITER_CNT), "run concatTwo32bitStreams");
	adfCheck(concatTwo32bitStreams.end(), "end concatTwo32bitStreams");

  // int8
  adfCheck(concatInt8.init(), "init concatInt8");
  adfCheck(concatInt8.run(ITER_CNT), "run concatInt8");
	adfCheck(concatInt8.end(), "end concatInt8");

  // stream
  adfCheck(concatInt8Stream2.init(), "init concatInt8Stream2");
  adfCheck(concatInt8Stream2.run(ITER_CNT), "run concatInt8Stream2");
	adfCheck(concatInt8Stream2.end(), "end concatInt8Stream2");
  
  adfCheck(concatInt8Stream3.init(), "init concatInt8Stream3");
  adfCheck(concatInt8Stream3.run(ITER_CNT), "run concatInt8Stream3");
	adfCheck(concatInt8Stream3.end(), "end concatInt8Stream3");

  adfCheck(concatInt8Stream4.init(), "init concatInt8Stream4");
  adfCheck(concatInt8Stream4.run(ITER_CNT), "run concatInt8Stream4");
	adfCheck(concatInt8Stream4.end(), "end concatInt8Stream4");
  
  adfCheck(concatInt8Stream5.init(), "init concatInt8Stream5");
  adfCheck(concatInt8Stream5.run(ITER_CNT), "run concatInt8Stream5");
	adfCheck(concatInt8Stream5.end(), "end concatInt8Stream5");

  adfCheck(concatInt8Stream6.init(), "init concatInt8Stream6");
  adfCheck(concatInt8Stream6.run(ITER_CNT), "run concatInt8Stream6");
	adfCheck(concatInt8Stream6.end(), "end concatInt8Stream6");

  adfCheck(concatInt8Stream7.init(), "init concatInt8Stream7");
  adfCheck(concatInt8Stream7.run(ITER_CNT), "run concatInt8Stream7");
	adfCheck(concatInt8Stream7.end(), "end concatInt8Stream7");

  adfCheck(concatInt8Stream8.init(), "init concatInt8Stream8");
  adfCheck(concatInt8Stream8.run(ITER_CNT), "run concatInt8Stream8");
	adfCheck(concatInt8Stream8.end(), "end concatInt8Stream8");
  

  // not very feasible mapping into array
  // adfCheck(multiConcatFloat.init(), "init multiConcatFloat");
  // adfCheck(multiConcatFloat.run(ITER_CNT), "run multiConcatFloat");
	// adfCheck(multiConcatFloat.end(), "end multiConcatFloat");
  return 0;
}
#endif
