#include "graph_qlinearmac.h"
#include "graph_utils.h"


template <template<int, int, int> class QLINEARMAC, 
  int B, int W, int IS_RELU>
class QlinearMacGraphTest : public adf::graph {

  private:
    QlinearMacGraph<QLINEARMAC, B, W, IS_RELU> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QlinearMacGraphTest(
      const std::string& id,
      std::vector<int8_t> weights,
      std::vector<int8_t> bias,
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t b_zero,
      int8_t z_zero,
      int8_t y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearmac_out.txt"
    ): g(weights, bias, x_scale, w_scale, b_scale, z_scale, y_scale, x_zero, w_zero, b_zero, z_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearmac"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearmac"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*W>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*W>> (g.pout[0], plout[0].in[0]);
    }
};

// chunk to reduce I/O buffer sizes
std::vector<int8_t> k2qmac_w {33, 87, 127, 44, 70, 9, 14, 78, 46, 115, 50, 26, 29, 24, 13, 23, 12, 24, 29, 62, 62, 43, 64, 15, 23, 43, 24, 41, 27, 17, 19, 15, 26, 48, 21, 42, 23, 59, 21, 67, 33, 21, 18, 21, 46, 15, 48, 39, 31, 16, 20, 19, 11, 22, 17, 36, 11, 33, 15, 69, 30, 32, 8, 10, 24, 29, 22, 31, 20, 21, 16, 86, 40, 7, 54, 27, 52, 69, 12, 52, 71, 16, 37, 16, 12, 9, 20, 39, 29, 11, 7, 58, 12, 51, 37, 42, 19, 45, 19, 31, 26, 44, 46, 17, 27, 28, 38, 37, 9, 16, 27, 27, 19, 32, 26, 27, 26, 19, 45, 23, 14, 18, 48, 47, 31, 35, 28, 15};
std::vector<int8_t> k2qmac_b {11, -8, -82, -35, 2, -42, -8, 89, -13, 2, 4, -3, -43, 1, 0, 13, 34, 19, 8, -14, -127, -3, 33, 17, 32, 83, -41, 29, -7, -16, -12, -7, -12, -38, 14, -9, 27, 20, -10, -29, 29, 2, -7, 5, -1, 0, -31, 8, 77, -20, -21, -12, -19, -23, -24, -58, -14, -9, -33, -6, 1, 17, -21, -56, -62, 13, -4, 8, -18, -12, -4, 30, 26, 40, 41, 22, -10, -3, -64, 20, -2, 9, -15, 7, 34, 35, 37, 4, 2, 18, 58, -2, 63, -87, 36, 6, -8, -22, -51, 17, 18, 83, 19, 15, 17, 27, -103, 12, 46, -25, 2, -18, 4, -9, -4, -10, -30, -2, 25, 26, -3, -2, 23, 0, 5, 42, -45, 52};
float_t k2qmac_xscale = 75.1787;
float_t k2qmac_wscale = 0.00017276738;
float_t k2qmac_bscale = 0.16271122;
float_t k2qmac_zscale = 0.2186131;
float_t k2qmac_yscale = 0.05708896;
int8_t k2qmac_xzero = 19;
int8_t k2qmac_wzero = 0;
int8_t k2qmac_bzero = 0;
int8_t k2qmac_zzero = -11;
int8_t k2qmac_yzero = -128;


// instance to be compiled and used in host within xclbin
QlinearMacGraphTest<QlinearMacScalar,14,128,0> qlinearmacScalar(
  "qlinearmacScalar", 
  k2qmac_w, k2qmac_b, 
  k2qmac_xscale, k2qmac_wscale, k2qmac_bscale, k2qmac_zscale, k2qmac_yscale, 
  k2qmac_xzero, k2qmac_wzero, k2qmac_bzero, k2qmac_zzero, k2qmac_yzero, 
  "k2qlinearmac_in_shape14x128.txt", 
  "k2qlinearmac_goldenout_shape14x128_scalar.txt");

QlinearMacGraphTest<QlinearMac,14,128,0> qlinearmac(
  "qlinearmac", 
  k2qmac_w, k2qmac_b, 
  k2qmac_xscale, k2qmac_wscale, k2qmac_bscale, k2qmac_zscale, k2qmac_yscale, 
  k2qmac_xzero, k2qmac_wzero, k2qmac_bzero, k2qmac_zzero, k2qmac_yzero, 
  "k2qlinearmac_in_shape14x128.txt", 
  "k2qlinearmac_goldenout_shape14x128.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	adfCheck(qlinearmacScalar.init(), "init qlinearmacScalar");
  adfCheck(qlinearmacScalar.run(ITER_CNT), "run qlinearmacScalar");
	adfCheck(qlinearmacScalar.end(), "end qlinearmacScalar");

  adfCheck(qlinearmac.init(), "init qlinearmac");
  adfCheck(qlinearmac.run(ITER_CNT), "run qlinearmac");
	adfCheck(qlinearmac.end(), "end qlinearmac");
  return 0;
}
#endif