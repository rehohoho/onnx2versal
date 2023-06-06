#include "graph_concat.h"
#include "graph_utils.h"


template <template<int, int, int, int> class CONCAT,
  int LCNT, int H, int INP_W, int OUT_W>
class ConcatGraphTest : public adf::graph {

  private:
    ConcatGraph<CONCAT, LCNT, H, INP_W, OUT_W> g;

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
        adf::connect<adf::window<H*INP_W*4>> (plin[i].out[0], g.pin[i]);
      }

      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<H*OUT_W*4>> (g.pout[0], plout[0].in[0]);
    }

};


template <template<int, int, int, int> class CONCAT,
  int LCNT, int H, int INP_W, int OUT_W>
class ConcatTwiceGraphTest : public adf::graph {

  private:
    ConcatTwiceGraph<CONCAT, LCNT, H, INP_W, OUT_W> g;

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
        adf::connect<adf::window<H*INP_W*4>> (plin[i].out[0], g.pin[i]);
      }
      
      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<H*OUT_W*4>> (g.pout[0], plout[0].in[0]);
    }

};

std::vector<std::string> inp_txts {
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt"
};

// instance to be compiled and used in host within xclbin
ConcatGraphTest<ConcatScalar, 5, 4, 16, 52> concatScalar(
  "concatScalar", "concat_fpout_shape4x52_ConcatScalar.txt", inp_txts);

ConcatGraphTest<ConcatVector, 5, 4, 16, 52> concatVector(
  "concatVector", "concat_fpout_shape4x52_ConcatVector.txt", inp_txts);

std::vector<std::string> multi_inp_txts {
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt"
};

ConcatTwiceGraphTest<ConcatVector, 52, 4, 16, 820> multiConcatVector(
  "multiConcatVector", "concatmulti_fpout_shape4x820_ConcatVector.txt", multi_inp_txts
);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(concatScalar.init(), "init concatScalar");
  adfCheck(concatScalar.run(ITER_CNT), "run concatScalar");
	adfCheck(concatScalar.end(), "end concatScalar");

  adfCheck(concatVector.init(), "init concatVector");
  adfCheck(concatVector.run(ITER_CNT), "run concatVector");
	adfCheck(concatVector.end(), "end concatVector");

  adfCheck(multiConcatVector.init(), "init multiConcatVector");
  adfCheck(multiConcatVector.run(ITER_CNT), "run multiConcatVector");
	adfCheck(multiConcatVector.end(), "end multiConcatVector");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(concatScalar.init(), "init concatScalar");
  get_graph_throughput_by_port(concatScalar, "plout[0]", concatScalar.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(concatScalar.end(), "end concatScalar");

  adfCheck(concatVector.init(), "init concatVector");
  get_graph_throughput_by_port(concatVector, "plout[0]", concatVector.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(concatVector.end(), "end concatVector");

  adfCheck(multiConcatVector.init(), "init multiConcatVector");
  get_graph_throughput_by_port(multiConcatVector, "plout[0]", multiConcatVector.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(multiConcatVector.end(), "end multiConcatVector");
  return 0;
}
#endif
