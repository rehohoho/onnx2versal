#include "graph_concat.h"
#include "graph_utils.h"


template <template<int, int, int, int> class CONCAT,
  int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
class ConcatGraphTest : public adf::graph {

  private:
    ConcatGraph<CONCAT, LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatGraphTest(
      const std::string& id,
      const std::string& OUT_TXT = "concat_out.txt",
      const std::string& INP0_TXT = std::string(),
      const std::string& INP1_TXT = std::string(),
      const std::string& INP2_TXT = std::string(),
      const std::string& INP3_TXT = std::string(),
      const std::string& INP4_TXT = std::string(),
      const std::string& INP5_TXT = std::string(),
      const std::string& INP6_TXT = std::string(),
      const std::string& INP7_TXT = std::string()
    ) {
#define SET_OPT_PLIN(TXT_PATH, i) \
      if (i < LCNT) { \
        std::string plin_name = "plin"+std::to_string(i)+"_concat_"+id+"_input"; \
        plin[i] = adf::input_plio::create(plin_name, PLIO64_ARG(TXT_PATH));}

      SET_OPT_PLIN(INP0_TXT, 0);
      SET_OPT_PLIN(INP1_TXT, 1);
      SET_OPT_PLIN(INP2_TXT, 2);
      SET_OPT_PLIN(INP3_TXT, 3);
      SET_OPT_PLIN(INP4_TXT, 4);
      SET_OPT_PLIN(INP5_TXT, 5);
      SET_OPT_PLIN(INP6_TXT, 6);
      SET_OPT_PLIN(INP7_TXT, 7);

      for (int i = 0; i < LCNT; i++)
        adf::connect<adf::window<CHUNK_CNT*CHUNK_SIZE*4>> (plin[i].out[0], g.pin[i]);
      
      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<CHUNK_CNT*BLOCK_SIZE*4>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
ConcatGraphTest<ConcatScalar, 5, 4, 16, 52> concatScalar(
  "concatScalar", "concat_fpout_shape4x52_ConcatScalar.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt"
);

ConcatGraphTest<ConcatVector, 5, 4, 16, 52> concatVector(
  "concatVector", "concat_fpout_shape4x52_ConcatVector.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt"
);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(concatScalar.init(), "init concatScalar");
  adfCheck(concatScalar.run(ITER_CNT), "run concatScalar");
	adfCheck(concatScalar.end(), "end concatScalar");

  adfCheck(concatVector.init(), "init concatVector");
  adfCheck(concatVector.run(ITER_CNT), "run concatVector");
	adfCheck(concatVector.end(), "end concatVector");
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
  return 0;
}
#endif
