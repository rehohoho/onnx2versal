#include "graph_qlinearsoftmax.h"
#include "graph_utils.h"


template <template<int, int, int> class QLINEARSOFTMAX, 
  int INP_H, int INP_W, int INP_W_PAD>
class QlinearsoftmaxGraphTest : public adf::graph {

  private:
    QlinearsoftmaxGraph<QLINEARSOFTMAX, INP_H, INP_W, INP_W_PAD> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QlinearsoftmaxGraphTest(
      const std::string& id,
      float x_scale,
      float y_scale,
      int8_t x_zero,
      int8_t y_zero,
      const std::string& INP_TXT, 
      const std::string& OUT_TXT
    ): g(x_scale, y_scale, x_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearsoftmax"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearsoftmax"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<INP_H*INP_W_PAD>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<INP_H*INP_W_PAD>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
QlinearsoftmaxGraphTest<QlinearsoftmaxScalar,10,20,32> qlinearsoftmaxScalar(
  "qlinearsoftmaxScalar", 
  0.004, 0.003, -128, -128,
  "qlinearsoftmax_int8in.txt", 
  "qlinearsoftmax_int8out_shape10x20_QlinearsoftmaxScalar.txt");

QlinearsoftmaxGraphTest<QlinearsoftmaxSingleaxis,10,20,32> qlinearsoftmaxSingleaxis(
  "qlinearsoftmaxSingleaxis", 
  0.004, 0.003, -128, -128,
  "qlinearsoftmax_int8in.txt", 
  "qlinearsoftmax_int8out_shape10x20_QlinearsoftmaxSingleaxis.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(qlinearsoftmaxScalar.init(), "init qlinearsoftmaxScalar");
  adfCheck(qlinearsoftmaxScalar.run(ITER_CNT), "run qlinearsoftmaxScalar");
	adfCheck(qlinearsoftmaxScalar.end(), "end qlinearsoftmaxScalar");

  adfCheck(qlinearsoftmaxSingleaxis.init(), "init qlinearsoftmaxSingleaxis");
  adfCheck(qlinearsoftmaxSingleaxis.run(ITER_CNT), "run qlinearsoftmaxSingleaxis");
	adfCheck(qlinearsoftmaxSingleaxis.end(), "end qlinearsoftmaxSingleaxis");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(qlinearsoftmaxScalar.init(), "init qlinearsoftmaxScalar");
  get_graph_throughput_by_port(qlinearsoftmaxScalar, "plout[0]", qlinearsoftmaxScalar.plout[0], 10*20, sizeof(int8_t), ITER_CNT);
	adfCheck(qlinearsoftmaxScalar.end(), "end qlinearsoftmaxScalar");

  adfCheck(qlinearsoftmaxSingleaxis.init(), "init qlinearsoftmaxSingleaxis");
  get_graph_throughput_by_port(qlinearsoftmaxSingleaxis, "plout[0]", qlinearsoftmaxSingleaxis.plout[0], 10*20, sizeof(int8_t), ITER_CNT);
	adfCheck(qlinearsoftmaxSingleaxis.end(), "end qlinearsoftmaxSingleaxis");
  return 0;
}
#endif
