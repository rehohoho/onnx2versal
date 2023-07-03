#include "graph_split.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitGraphTest : public adf::graph {

  private:
    typedef SplitGraph<SPLIT, TT, H, INP_W, OUT_W, OVERLAP> Graph;
    Graph g;
    static constexpr int LCNT = Graph::LCNT;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[LCNT];

    SplitGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::vector<std::string> OUT_TXT = std::vector<std::string>()
    ) {
      plin[0] = adf::input_plio::create("plin0_split_"+id+"_input", PLIO64_ARG(INP_TXT));      
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);

      for (int i = 0; i < LCNT; i++) {
        plout[i] = adf::output_plio::create("plout"+std::to_string(i)+"_split_"+id+"_output", PLIO64_ARG(OUT_TXT[i]));
        adf::connect<adf::window<H*OUT_W*sizeof(TT)>> (g.pout[i], plout[i].in[0]);
      }
    }

};


template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitTwoStreamGraphTest : public adf::graph {

  private:
    SplitTwoStreamGraph<SPLIT, TT, H, INP_W, OUT_W, OVERLAP> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[2];

    SplitTwoStreamGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT0,
      const std::string& OUT_TXT1
    ) {
      plin[0] = adf::input_plio::create("plin0_split_"+id+"_input", PLIO64_ARG(INP_TXT));      
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);

      plout[0] = adf::output_plio::create("plout0_split_"+id+"_output", PLIO64_ARG(OUT_TXT0));
      plout[1] = adf::output_plio::create("plout1_split_"+id+"_output", PLIO64_ARG(OUT_TXT1));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
      adf::connect<adf::stream> (g.pout[1], plout[1].in[0]);
    }

};


template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitFilterStreamGraphTest : public adf::graph {

  private:
    SplitFilterStreamGraph<SPLIT, TT, H, INP_W, OUT_W, OVERLAP> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    SplitFilterStreamGraphTest(
      const std::string& id,
      int lane_idx,
      const std::string& INP_TXT,
      const std::string& OUT_TXT0
    ): g(lane_idx) {
      plin[0] = adf::input_plio::create("plin0_split_"+id+"_input", PLIO64_ARG(INP_TXT));      
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);

      plout[0] = adf::output_plio::create("plout0_split_"+id+"_output", PLIO64_ARG(OUT_TXT0));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitFilterStreamTwiceGraphTest : public adf::graph {

  private:
    SplitFilterStreamTwiceGraph<SPLIT, TT, H, INP_W, OUT_W, OVERLAP> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[2];

    SplitFilterStreamTwiceGraphTest(
      const std::string& id,
      int lane_idx,
      const std::string& INP_TXT,
      const std::string& OUT_TXT0,
      const std::string& OUT_TXT1
    ): g(lane_idx) {
      plin[0] = adf::input_plio::create("plin0_split_"+id+"_input", PLIO64_ARG(INP_TXT));      
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);

      plout[0] = adf::output_plio::create("plout0_split_"+id+"_output", PLIO64_ARG(OUT_TXT0));
      plout[1] = adf::output_plio::create("plout1_split_"+id+"_output", PLIO64_ARG(OUT_TXT1));
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
      adf::connect<adf::stream> (g.pout[1], plout[1].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
const int H = 10;
const int INP_W = 64;
const int OUT_W = 22;
const int OVERLAP = 1;

// float
SplitGraphTest<SplitScalar, float_t, H, INP_W, OUT_W, OVERLAP> splitScalar(
  "splitScalar", "split_fpin.txt", 
  {
    "split_fpout0_shape10x22_splitScalar.txt", 
    "split_fpout1_shape10x22_splitScalar.txt",
    "split_fpout2_shape10x22_splitScalar.txt"
  });

SplitTwoStreamGraphTest<SplitTwo32bitStreams, float, H, INP_W, OUT_W, OVERLAP> splitTwo32bitStreams(
  "splitTwo32bitStreams", "split_fpin.txt", 
  "split_fpout0_2stream_shape10x44_SplitTwo32bitStreams.txt", "split_fpout1_2stream_shape10x22_SplitTwo32bitStreams.txt");

SplitFilterStreamGraphTest<SplitFilterFloatStream, float, H, INP_W, OUT_W, OVERLAP> splitFilterFloatStream(
  "splitFilterFloatStream", 1,
  "split_fpin.txt", "split_fpout1_shape10x22_SplitFilterFloatStream.txt");

SplitFilterStreamTwiceGraphTest<SplitFilterFloatStreamTwice, float, H, INP_W, OUT_W, OVERLAP> splitFilterFloatStreamTwice(
  "splitFilterFloatStreamTwiceTwice", 1,
  "split_fpin.txt", 
  "split_fpout1_shape10x22_SplitFilterFloatStreamTwice.txt",
  "split_fpout2_shape10x22_SplitFilterFloatStreamTwice.txt");

SplitGraphTest<SplitScalar, float_t, H, INP_W, 31, -1> splitScalar_neg(
  "splitScalar_neg", "split_fpin.txt", 
  {
    "split_fpout0_shape10x31_splitScalar.txt", 
    "split_fpout1_shape10x31_splitScalar.txt"
  });

SplitTwoStreamGraphTest<SplitTwo32bitStreams, float, H, INP_W, 31, -1> splitTwo32bitStreams_neg(
  "splitTwo32bitStreams_neg", "split_fpin.txt", 
  "split_fpout0_neg2stream_shape10x31_SplitTwo32bitStreams.txt", "split_fpout1_neg2stream_shape10x31_SplitTwo32bitStreams.txt");


// int8
const int INP_W_INT8 = 160;
const int OUT_W_INT8 = 64;
const int OVERLAP_INT8 = 16;

SplitGraphTest<SplitInt8, int8_t, H, INP_W_INT8, OUT_W_INT8, OVERLAP_INT8> splitInt8(
  "splitInt8", "split_int8in_shape10x160.txt", 
  {
    "split_int8out0_shape10x64_splitInt8.txt", 
    "split_int8out1_shape10x64_splitInt8.txt",
    "split_int8out2_shape10x64_splitInt8.txt"
  });

SplitGraphTest<SplitInt8, int8_t, H, INP_W_INT8, 64, -32> splitInt8_neg(
  "splitInt8_neg", "split_int8in_shape10x160.txt", 
  {
    "split_int8out0_neg_shape10x64_splitInt8.txt", 
    "split_int8out1_neg_shape10x64_splitInt8.txt"
  });


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // float
  adfCheck(splitScalar.init(), "init splitScalar");
  adfCheck(splitScalar.run(ITER_CNT), "run splitScalar");
	adfCheck(splitScalar.end(), "end splitScalar");

  adfCheck(splitTwo32bitStreams.init(), "init splitTwo32bitStreams");
  adfCheck(splitTwo32bitStreams.run(ITER_CNT), "run splitTwo32bitStreams");
	adfCheck(splitTwo32bitStreams.end(), "end splitTwo32bitStreams");

  adfCheck(splitFilterFloatStream.init(), "init splitFilterFloatStream");
  adfCheck(splitFilterFloatStream.run(ITER_CNT), "run splitFilterFloatStream");
	adfCheck(splitFilterFloatStream.end(), "end splitFilterFloatStream");

  adfCheck(splitFilterFloatStreamTwice.init(), "init splitFilterFloatStreamTwice");
  adfCheck(splitFilterFloatStreamTwice.run(ITER_CNT), "run splitFilterFloatStreamTwice");
	adfCheck(splitFilterFloatStreamTwice.end(), "end splitFilterFloatStreamTwice");

  adfCheck(splitScalar_neg.init(), "init splitScalar_neg");
  adfCheck(splitScalar_neg.run(ITER_CNT), "run splitScalar_neg");
	adfCheck(splitScalar_neg.end(), "end splitScalar_neg");

  adfCheck(splitTwo32bitStreams_neg.init(), "init splitTwo32bitStreams_neg");
  adfCheck(splitTwo32bitStreams_neg.run(ITER_CNT), "run splitTwo32bitStreams_neg");
	adfCheck(splitTwo32bitStreams_neg.end(), "end splitTwo32bitStreams_neg");

  // int8
  adfCheck(splitInt8.init(), "init splitInt8");
  adfCheck(splitInt8.run(ITER_CNT), "run splitInt8");
	adfCheck(splitInt8.end(), "end splitInt8");

  adfCheck(splitInt8_neg.init(), "init splitInt8_neg");
  adfCheck(splitInt8_neg.run(ITER_CNT), "run splitInt8_neg");
	adfCheck(splitInt8_neg.end(), "end splitInt8_neg");
  return 0;
}
#endif
