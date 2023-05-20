#include "graph_qlinearconv.h"
#include "graph_utils.h"


template <template<int, int, int, int, int, int, int, int> class CONV, 
  int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
class QLinearConvGraphTest : public adf::graph {

  private:
    QLinearConvGraph<CONV, INP_H, INP_W, OUT_H, OUT_W, B, C, M, K> g;

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
      adf::connect<adf::window<B*INP_H*INP_W*C>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_H*OUT_W*M>> (g.pout[0], plout[0].in[0]);
    }
};

// instance to be compiled and used in host within xclbin
std::vector<int8_t> fpweights {127, 127, 127, 127, 127, 127, 127, 127};
std::vector<int32_t> fpbias {146, 146, 146, 146, 146, 146, 146, 146};

std::vector<int8_t> lenet_w {20, 108, 122, 127, 88, 16, 83, 95, 123, 14, 48, 33, 3, 18, 44, -6, 10, -34, -3, -16, -5, 6, -29, -58, -13, 63, -23, 27, 35, 85, 48, 45, -10, 73, 38, -28, -13, -5, 2, -20, -13, -42, -19, -8, 9, -48, -28, -37, 4, -27, -16, -27, 55, 6, 2, 6, -7, -2, 34, -19, 79, 82, 7, 49, 58, 43, 110, 78, 23, -28, 87, 66, 26, 0, 34, -36, 31, -6, 13, -8, -24, 51, 30, 0, -53, 58, 20, 19, 4, -16, 39, 97, 69, 68, 1, 16, 83, 105, 24, 84, -7, -26, 16, 34, -1, -49, -34, 15, -4, 29, 53, -28, 51, 44, 76, 35, 43, 23, 5, 66, -28, 49, -14, -39, -11, 19, -55, -53, 6, 6, 21, -26, -35, -51, 22, -2, -4, -27, -42, 8, 73, 2, -15, 37, 61, 26, 97, 75, 102, 31};
std::vector<int32_t> lenet_b {924, 305, 4023, 252, 4657, 9696};

//BCHW
QLinearConvGraphTest<QLinearConvScalar, 28, 32, 28, 32, 1, 1, 8, 1> qLinearConvScalar(
  "qLinearConvScalar", fpweights, fpbias, 0.00369204697, 0.003, 0.00162681262, 25, 0, 19,
  "qlinearconv_int8in.txt", "qlinearconv_int8out_QLinearConvScalar.txt");

QLinearConvGraphTest<QLinearConvScalar, 28, 32, 24, 32, 1, 1, 6, 5> lenet(
  "lenet", lenet_w, lenet_b, 0.00392157, 0.00415155, 0.01521751, -128, 0, -128,
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
