#include "graph_argmax.h"
#include "graph_utils.h"

#define ITER_CNT 1


template <template<int, int> class ARGMAX, int WINDOW_SIZE, int CHUNK_SIZE>
class ArgmaxGraphTest : public adf::graph {

  private:
    ArgmaxGraph<ARGMAX, WINDOW_SIZE, CHUNK_SIZE> g;

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
      adf::connect<adf::window<WINDOW_SIZE*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<WINDOW_SIZE/CHUNK_SIZE*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
ArgmaxGraphTest<ArgmaxScalar, 100, 10> fpscalar(
  "fpscalar", "argmax_fpin.txt", "argmax_fpout_ArgmaxScalar.txt");
ArgmaxGraphTest<ArgmaxScalar, 100, 10> fpscalar_rand(
  "fpscalar_rand", "argmax_fpin_rand.txt", "argmax_fpout_ArgmaxScalar_rand.txt");



#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(fpscalar.init(), "init fpscalar");
  adfCheck(fpscalar.run(1), "run fpscalar");
	adfCheck(fpscalar.end(), "end fpscalar");

  adfCheck(fpscalar_rand.init(), "init fpscalar_rand");
  adfCheck(fpscalar_rand.run(1), "run fpscalar_rand");
	adfCheck(fpscalar_rand.end(), "end fpscalar_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(fpscalar.init(), "init fpscalar");
  get_graph_throughput_by_port(fpscalar, "plout[0]", fpscalar.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(fpscalar.end(), "end fpscalar");

  adfCheck(fpscalar_rand.init(), "init fpscalar_rand");
  get_graph_throughput_by_port(fpscalar_rand, "plout[0]", fpscalar_rand.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(fpscalar_rand.end(), "end fpscalar_rand");
  return 0;
}
#endif
