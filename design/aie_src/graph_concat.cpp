#include "graph_concat.h"
#include "graph_utils.h"

#define ITER_CNT 1


template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class ConcatScalarGraphTest : public adf::graph {

  private:
    ConcatScalarGraph<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatScalarGraphTest(
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
        adf::connect<adf::window<WINDOW_SIZE*4>> (plin[i].out[0], g.pin[i]);
      
      // BLOCK_SIZE <= WINDOW_SIZE
      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<WINDOW_SIZE/CHUNK_SIZE*BLOCK_SIZE*4>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
ConcatScalarGraphTest<5, 8, 8, 4*8+4> fpscalar1(
  "fpscalar1", "concat_fpout1_ConcatScalar.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt"
);

ConcatScalarGraphTest<5, 8, 4, 4*4+2> fpscalar2(
  "fpscalar2", "concat_fpout2_ConcatScalar.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt", "concat_fpin.txt",
  "concat_fpin.txt"
);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(fpscalar1.init(), "init fpscalar1");
  adfCheck(fpscalar1.run(1), "run fpscalar1");
	adfCheck(fpscalar1.end(), "end fpscalar1");

  adfCheck(fpscalar2.init(), "init fpscalar2");
  adfCheck(fpscalar2.run(1), "run fpscalar2");
	adfCheck(fpscalar2.end(), "end fpscalar2");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(fpscalar1.init(), "init fpscalar1");
  get_graph_throughput_by_port(fpscalar1, "plout[0]", fpscalar1.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(fpscalar1.end(), "end fpscalar1");

  adfCheck(fpscalar2.init(), "init fpscalar2");
  get_graph_throughput_by_port(fpscalar2, "plout[0]", fpscalar2.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(fpscalar2.end(), "end fpscalar2");
  return 0;
}
#endif
