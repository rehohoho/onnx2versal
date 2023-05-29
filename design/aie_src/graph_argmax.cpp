#include "graph_argmax.h"
#include "graph_utils.h"


template <template<int, int> class ARGMAX, int CHUNK_CNT, int CHUNK_SIZE>
class ArgmaxGraphTest : public adf::graph {

  private:
    ArgmaxGraph<ARGMAX, CHUNK_CNT, CHUNK_SIZE> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    ArgmaxGraphTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ) { 
      plin[0] = adf::input_plio::create("plin0_argmax"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_argmax"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<CHUNK_CNT*CHUNK_SIZE*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<CHUNK_CNT*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
// padded to vector boundary
ArgmaxGraphTest<ArgmaxScalar, 10, 10> argmaxScalar(
  "argmaxScalar", "argmax_fpin.txt", "argmax_fpout_shape1x10_ArgmaxScalar.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(argmaxScalar.init(), "init argmaxScalar");
  adfCheck(argmaxScalar.run(ITER_CNT), "run argmaxScalar");
	adfCheck(argmaxScalar.end(), "end argmaxScalar");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(argmaxScalar.init(), "init argmaxScalar");
  get_graph_throughput_by_port(argmaxScalar, "plout[0]", argmaxScalar.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(argmaxScalar.end(), "end argmaxScalar");
  return 0;
}
#endif
