#include "graph_softmax.h"
#include "graph_utils.h"


template <template<int, int> class SOFTMAX, int CHUNK_CNT, int CHUNK_SIZE>
class SoftmaxGraphTest : public adf::graph {

  private:
    SoftmaxGraph<SOFTMAX, CHUNK_CNT, CHUNK_SIZE> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    SoftmaxGraphTest(
      const std::string& id,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ) { 
      plin[0] = adf::input_plio::create("plin0_softmax"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_softmax"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<CHUNK_CNT*CHUNK_SIZE*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<CHUNK_CNT*CHUNK_SIZE*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
// padded to vector boundary
SoftmaxGraphTest<SoftmaxScalar, 10, 10> softmaxScalar(
  "softmaxScalar", "softmax_fpin.txt", "softmax_fpout_shape10x10_SoftmaxScalar.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(softmaxScalar.init(), "init softmaxScalar");
  adfCheck(softmaxScalar.run(ITER_CNT), "run softmaxScalar");
	adfCheck(softmaxScalar.end(), "end softmaxScalar");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(softmaxScalar.init(), "init softmaxScalar");
  get_graph_throughput_by_port(softmaxScalar, "plout[0]", softmaxScalar.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(softmaxScalar.end(), "end softmaxScalar");
  return 0;
}
#endif
