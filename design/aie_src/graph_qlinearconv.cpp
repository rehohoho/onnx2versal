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
std::vector<int8_t> fpweights {0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0, 0, 0, 0, 0, 0, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 0, 0, 0, 0, 0, 0, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 0, 0, 0, 0, 0, 0, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 0, 0, 0, 0, 0, 0, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 0, 0, 0, 0, 0, 0, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 0, 0, 0, 0, 0, 0, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 0, 0, 0, 0, 0, 0, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 0, 0, 0, 0, 0, 0, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 0, 0, 0, 0, 0, 0, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 0, 0, 0, 0, 0, 0, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 0, 0, 0, 0, 0, 0, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 0, 0, 0, 0, 0, 0, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 0, 0, 0, 0, 0, 0, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 0, 0, 0, 0, 0, 0, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 0, 0, 0, 0, 0, 0, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 0, 0, 0, 0, 0, 0, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 0, 0, 0, 0, 0, 0, 90, 90, 91, 91, 92, 92, 93, 93, 94, 94, 0, 0, 0, 0, 0, 0, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 0, 0, 0, 0, 0, 0, 100, 100, 101, 101, 102, 102, 103, 103, 104, 104, 0, 0, 0, 0, 0, 0, 105, 105, 106, 106, 107, 107, 108, 108, 109, 109, 0, 0, 0, 0, 0, 0, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 0, 0, 0, 0, 0, 0, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 0, 0, 0, 0, 0, 0, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 0, 0, 0, 0, 0, 0, 125, 125, 126, 126, 127, 127, -128, -128, -127, -127, 0, 0, 0, 0, 0, 0, -126, -126, -125, -125, -124, -124, -123, -123, -122, -122, 0, 0, 0, 0, 0, 0, -121, -121, -120, -120, -119, -119, -118, -118, -117, -117, 0, 0, 0, 0, 0, 0, -116, -116, -115, -115, -114, -114, -113, -113, -112, -112, 0, 0, 0, 0, 0, 0, -111, -111, -110, -110, -109, -109, -108, -108, -107, -107, 0, 0};
std::vector<int32_t> fpbias {0, 166, 333, 500, 666, 833};

std::vector<int8_t> lenet_w {0, 0, 0, 0, 20, 20, 108, 108, 122, 122, 127, 127, 88, 88, 0, 0, 0, 0, 0, 0, 16, 16, 83, 83, 95, 95, 123, 123, 14, 14, 0, 0, 0, 0, 0, 0, 48, 48, 33, 33, 3, 3, 18, 18, 44, 44, 0, 0, 0, 0, 0, 0, -6, -6, 10, 10, -34, -34, -3, -3, -16, -16, 0, 0, 0, 0, 0, 0, -5, -5, 6, 6, -29, -29, -58, -58, -13, -13, 0, 0, 0, 0, 0, 0, 63, 63, -23, -23, 27, 27, 35, 35, 85, 85, 0, 0, 0, 0, 0, 0, 48, 48, 45, 45, -10, -10, 73, 73, 38, 38, 0, 0, 0, 0, 0, 0, -28, -28, -13, -13, -5, -5, 2, 2, -20, -20, 0, 0, 0, 0, 0, 0, -13, -13, -42, -42, -19, -19, -8, -8, 9, 9, 0, 0, 0, 0, 0, 0, -48, -48, -28, -28, -37, -37, 4, 4, -27, -27, 0, 0, 0, 0, 0, 0, -16, -16, -27, -27, 55, 55, 6, 6, 2, 2, 0, 0, 0, 0, 0, 0, 6, 6, -7, -7, -2, -2, 34, 34, -19, -19, 0, 0, 0, 0, 0, 0, 79, 79, 82, 82, 7, 7, 49, 49, 58, 58, 0, 0, 0, 0, 0, 0, 43, 43, 110, 110, 78, 78, 23, 23, -28, -28, 0, 0, 0, 0, 0, 0, 87, 87, 66, 66, 26, 26, 0, 0, 34, 34, 0, 0, 0, 0, 0, 0, -36, -36, 31, 31, -6, -6, 13, 13, -8, -8, 0, 0, 0, 0, 0, 0, -24, -24, 51, 51, 30, 30, 0, 0, -53, -53, 0, 0, 0, 0, 0, 0, 58, 58, 20, 20, 19, 19, 4, 4, -16, -16, 0, 0, 0, 0, 0, 0, 39, 39, 97, 97, 69, 69, 68, 68, 1, 1, 0, 0, 0, 0, 0, 0, 16, 16, 83, 83, 105, 105, 24, 24, 84, 84, 0, 0, 0, 0, 0, 0, -7, -7, -26, -26, 16, 16, 34, 34, -1, -1, 0, 0, 0, 0, 0, 0, -49, -49, -34, -34, 15, 15, -4, -4, 29, 29, 0, 0, 0, 0, 0, 0, 53, 53, -28, -28, 51, 51, 44, 44, 76, 76, 0, 0, 0, 0, 0, 0, 35, 35, 43, 43, 23, 23, 5, 5, 66, 66, 0, 0, 0, 0, 0, 0, -28, -28, 49, 49, -14, -14, -39, -39, -11, -11, 0, 0, 0, 0, 0, 0, 19, 19, -55, -55, -53, -53, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 21, 21, -26, -26, -35, -35, -51, -51, 22, 22, 0, 0, 0, 0, 0, 0, -2, -2, -4, -4, -27, -27, -42, -42, 8, 8, 0, 0, 0, 0, 0, 0, 73, 73, 2, 2, -15, -15, 37, 37, 61, 61, 0, 0, 0, 0, 0, 0, 26, 26, 97, 97, 75, 75, 102, 102, 31, 31, 0, 0};
std::vector<int32_t> lenet_b {924, 305, 4023, 252, 4657, 9696};

QLinearConvGraphTest<QLinearConvScalar, 28, 32, 24, 32, 1, 1, 6, 5> qLinearConvScalar(
  "qLinearConvScalar", fpweights, fpbias, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in.txt", "qlinearconv_int8out_QLinearConvScalar.txt");

QLinearConvGraphTest<QLinearConvVector, 28, 32, 24, 32, 1, 1, 6, 5> qLinearConvVector(
  "qLinearConvVector", fpweights, fpbias, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in.txt", "qlinearconv_int8out_QLinearConvVector.txt");

QLinearConvGraphTest<QLinearConvVector, 28, 32, 24, 32, 1, 1, 6, 5> lenet(
  "lenet", lenet_w, lenet_b, 0.00392157, 0.00415155, 0.01544408, -128, 0, -128,
  "k1qlinearconv_in.txt", "k1qlinearconv_goldenout.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(qLinearConvScalar.init(), "init qLinearConvScalar");
  adfCheck(qLinearConvScalar.run(ITER_CNT), "run qLinearConvScalar");
	adfCheck(qLinearConvScalar.end(), "end qLinearConvScalar");

  adfCheck(qLinearConvVector.init(), "init qLinearConvVector");
  adfCheck(qLinearConvVector.run(ITER_CNT), "run qLinearConvVector");
	adfCheck(qLinearConvVector.end(), "end qLinearConvVector");

  adfCheck(lenet.init(), "init lenet");
  adfCheck(lenet.run(ITER_CNT), "run lenet");
	adfCheck(lenet.end(), "end lenet");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  adfCheck(qLinearConvScalar.init(), "init qLinearConvScalar");
  get_graph_throughput_by_port(qLinearConvScalar, "plout[0]", qLinearConvScalar.plout[0], 1*6*32*32, sizeof(int8_t), ITER_CNT);
	adfCheck(qLinearConvScalar.end(), "end qLinearConvScalar");

  adfCheck(qLinearConvVector.init(), "init qLinearConvVector");
  get_graph_throughput_by_port(qLinearConvVector, "plout[0]", qLinearConvVector.plout[0], 1*6*32*32, sizeof(int8_t), ITER_CNT);
	adfCheck(qLinearConvVector.end(), "end qLinearConvVector");

  adfCheck(lenet.init(), "init lenet");
  get_graph_throughput_by_port(lenet, "plout[0]", lenet.plout[0], 1*6*32*32, sizeof(int8_t), ITER_CNT);
	adfCheck(lenet.end(), "end lenet");
  return 0;
}
#endif
