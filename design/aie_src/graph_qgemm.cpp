#include "graph_qgemm.h"
#include "graph_utils.h"


template <template<int, int, int> class QGEMM, int M, int K, int N>
class QgemmGraphTest : public adf::graph {

  private:
    QgemmGraph<QGEMM, M, K, N> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QgemmGraphTest(
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
      const std::string& OUT_TXT = "qgemm_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero_point, w_zero_point, y_zero_point) { 
      plin[0] = adf::input_plio::create("plin0_qgemm"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qgemm"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<M*K>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<M*N>> (g.pout[0], plout[0].in[0]);
    }

};

// instance to be compiled and used in host within xclbin
// mknk has no padding, mkkn pads N in 128-bit chunks for vector kernel
std::vector<int8_t> weights {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0};
std::vector<int32_t> bias {0, 166, 333, 500, 666, 833, 1000, 1166, 1333, 1500, 0, 0, 0, 0, 0, 0};

QgemmGraphTest<QgemmScalar, 1, 84, 16> qgemmScalar(
  "qgemmScalar", weights, bias, 0.004, 0.003, 0.002, 25, 0, 19,
  "qgemm_int8in.txt", "qgemm_int8out_shape1x10_qgemmScalar.txt");

QgemmGraphTest<QgemmVector, 1, 84, 16> qgemmVector(
  "qgemmVector", weights, bias, 0.004, 0.003, 0.002, 25, 0, 19,
  "qgemm_int8in.txt", "qgemm_int8out_shape1x10_qgemmVector.txt");

#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(qgemmScalar.init(), "init qgemmScalar");
  adfCheck(qgemmScalar.run(ITER_CNT), "run qgemmScalar");
	adfCheck(qgemmScalar.end(), "end qgemmScalar");

  adfCheck(qgemmVector.init(), "init qgemmVector");
  adfCheck(qgemmVector.run(ITER_CNT), "run qgemmVector");
	adfCheck(qgemmVector.end(), "end qgemmVector");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  adfCheck(qgemmScalar.init(), "init qgemmScalar");
  get_graph_throughput_by_port(qgemmScalar, "plout[0]", qgemmScalar.plout[0], 1*16, sizeof(int8_t), ITER_CNT);
	adfCheck(qgemmScalar.end(), "end qgemmScalar");

  adfCheck(qgemmVector.init(), "init qgemmVector");
  get_graph_throughput_by_port(qgemmVector, "plout[0]", qgemmVector.plout[0], 1*16, sizeof(int8_t), ITER_CNT);
	adfCheck(qgemmVector.end(), "end qgemmVector");
  return 0;
}
#endif
