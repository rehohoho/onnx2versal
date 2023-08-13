#include "graph_softmax.h"
#include "graph_utils.h"


template <template<int, int, int> class SOFTMAX,
  int INP_H, int INP_W, int INP_W_PAD>
class SoftmaxGraphTest : public adf::graph {

  private:
    SoftmaxGraph<SOFTMAX, INP_H, INP_W, INP_W_PAD> g;

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
      adf::connect<adf::window<INP_H*INP_W_PAD*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<INP_H*INP_W_PAD*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
SoftmaxGraphTest<SoftmaxScalar, 10, 10, 16> softmaxScalar(
  "softmaxScalar", "softmax_fpin.txt", "softmax_fpout_shape10x10_SoftmaxScalar.txt");
SoftmaxGraphTest<SoftmaxSingleaxis, 10, 10, 16> softmaxSingleaxis(
  "softmaxSingleaxis", "softmax_fpin.txt", "softmax_fpout_shape10x10_SoftmaxSingleaxis.txt");
SoftmaxGraphTest<SoftmaxMultiaxis, 10, 10, 16> softmaxMultiaxis(
  "softmaxMultiaxis", "softmax_fpin.txt", "softmax_fpout_shape10x10_SoftmaxMultiaxis.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	adfCheck(softmaxScalar.init(), "init softmaxScalar");
  adfCheck(softmaxScalar.run(ITER_CNT), "run softmaxScalar");
	adfCheck(softmaxScalar.end(), "end softmaxScalar");

  adfCheck(softmaxSingleaxis.init(), "init softmaxSingleaxis");
  adfCheck(softmaxSingleaxis.run(ITER_CNT), "run softmaxSingleaxis");
	adfCheck(softmaxSingleaxis.end(), "end softmaxSingleaxis");

  adfCheck(softmaxMultiaxis.init(), "init softmaxMultiaxis");
  adfCheck(softmaxMultiaxis.run(ITER_CNT), "run softmaxMultiaxis");
	adfCheck(softmaxMultiaxis.end(), "end softmaxMultiaxis");
  return 0;
}
#endif
