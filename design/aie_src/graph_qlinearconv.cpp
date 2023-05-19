#include "graph_qlinearconv.h"
#include "graph_utils.h"


template <template<int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K>
class QLinearConvGraphTest : public adf::graph {

  private:
    QLinearConvGraph<CONV, INP_W, OUT_W, B, C, M, K> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QLinearConvGraphTest(
      const std::string& id,
      std::vector<int8_t> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero_point,
      int8_t w_zero_point,
      int8_t y_zero_point,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearconv_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero_point, w_zero_point, y_zero_point) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_W*INP_W*C>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M>> (g.pout[0], plout[0].in[0]);
    }
};

// instance to be compiled and used in host within xclbin
std::vector<int8_t> fpweights {127, 127, 127, 127, 127, 127, 127, 127};
std::vector<int32_t> fpbias {146, 146, 146, 146, 146, 146, 146, 146};

std::vector<int8_t> lenet_w {-17, -50, 1, 9, 127, 1};
std::vector<int32_t> lenet_b {-9955, 8150, -5166, 9124, 8301, -5062};

//BCHW
QLinearConvGraphTest<QLinearConvScalar, 28, 28, 1, 1, 8, 1> qLinearConvScalar(
  "qLinearConvScalar", fpweights, fpbias, 0.00369204697, 0.003, 0.00162681262, 25, 0, 19,
  "qlinearconv_int8in.txt", "qlinearconv_int8out_QLinearConvScalar.txt");

QLinearConvGraphTest<QLinearConvScalar, 28, 28, 1, 1, 6, 1> lenet(
  "lenet", lenet_w, lenet_b, 0.00392157, 0.01691757, 0.01058531, -128, 0, -128,
  "k1qlinearconv_in.txt", "k1qlinearconv_goldenout.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(qLinearConvScalar.init(), "init qLinearConvScalar");
  adfCheck(qLinearConvScalar.run(ITER_CNT), "run qLinearConvScalar");
	adfCheck(qLinearConvScalar.end(), "end qLinearConvScalar");

  adfCheck(lenet.init(), "init lenet");
  adfCheck(lenet.run(ITER_CNT), "run lenet");
	adfCheck(lenet.end(), "end lenet");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  // BCHW
  adfCheck(qLinearConvScalar.init(), "init qLinearConvScalar");
  get_graph_throughput_by_port(qLinearConvScalar, "plout[0]", qLinearConvScalar.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(qLinearConvScalar.end(), "end qLinearConvScalar");
  return 0;
}
#endif
