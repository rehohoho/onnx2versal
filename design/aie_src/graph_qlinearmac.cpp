#include "graph_qlinearmac.h"
#include "graph_utils.h"


template <
  template<typename, typename, int, int, int> class QLINEARMAC, 
  typename TT, typename TTPARAM, int B, int W, int IS_RELU>
class QlinearMacGraphTest : public adf::graph {

  private:
    QlinearMacGraph<QLINEARMAC, TT, TTPARAM, B, W, IS_RELU> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QlinearMacGraphTest(
      const std::string& id,
      std::vector<TTPARAM> weights,
      std::vector<TTPARAM> bias,
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TTPARAM b_zero,
      TT z_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearmac_out.txt"
    ): g(weights, bias, x_scale, w_scale, b_scale, z_scale, y_scale, x_zero, w_zero, b_zero, z_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearmac"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearmac"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, typename, int, int, int> class QLINEARMAC, 
  typename TT, typename TTPARAM, int B, int W, int IS_RELU>
class QlinearMacStreamGraphTest : public adf::graph {

  private:
    QlinearMacStreamGraph<QLINEARMAC, TT, TTPARAM, B, W, IS_RELU> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QlinearMacStreamGraphTest(
      const std::string& id,
      std::vector<TTPARAM> weights,
      std::vector<TTPARAM> bias,
      float x_scale,
      float w_scale,
      float b_scale,
      float z_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TTPARAM b_zero,
      TT z_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearmac_out.txt"
    ): g(weights, bias, x_scale, w_scale, b_scale, z_scale, y_scale, x_zero, w_zero, b_zero, z_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearmac"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearmac"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// chunk to reduce I/O buffer sizes
std::vector<uint8_t> k002qlinearmac_w {67, 175, 255, 89, 141, 19, 27, 156, 92, 231, 100, 51, 59, 48, 25, 46, 25, 48, 58, 125, 125, 86, 129, 30, 47, 86, 49, 83, 55, 35, 39, 30, 52, 96, 41, 85, 47, 118, 42, 134, 66, 43, 36, 43, 93, 31, 97, 79, 62, 32, 39, 38, 22, 44, 34, 73, 21, 66, 31, 138, 61, 65, 16, 20, 48, 59, 45, 63, 40, 42, 32, 173, 80, 14, 108, 55, 104, 139, 23, 105, 143, 33, 75, 32, 23, 18, 40, 78, 58, 22, 15, 117, 24, 103, 74, 84, 38, 91, 38, 61, 52, 89, 93, 34, 54, 55, 76, 74, 18, 32, 54, 54, 37, 64, 51, 53, 53, 39, 91, 47, 28, 36, 97, 95, 62, 71, 57, 30};
std::vector<uint8_t> k002qlinearmac_b {163, 140, 53, 109, 152, 100, 141, 255, 135, 152, 154, 146, 99, 151, 151, 166, 190, 172, 160, 134, 0, 147, 189, 170, 187, 248, 102, 184, 142, 131, 135, 142, 135, 105, 167, 140, 181, 173, 138, 115, 185, 152, 142, 155, 149, 150, 114, 159, 241, 126, 125, 136, 128, 122, 122, 82, 133, 139, 112, 142, 151, 170, 125, 84, 76, 165, 145, 159, 129, 136, 145, 186, 180, 198, 199, 176, 138, 147, 74, 173, 147, 161, 132, 158, 190, 192, 193, 154, 153, 171, 218, 148, 225, 47, 193, 157, 140, 124, 90, 170, 171, 248, 172, 168, 170, 182, 29, 164, 204, 120, 153, 129, 155, 139, 145, 138, 115, 148, 180, 181, 146, 148, 178, 150, 156, 199, 97, 211};
float_t k002qlinearmac_xscale = 65.746216;
float_t k002qlinearmac_wscale = 8.604493e-05;
float_t k002qlinearmac_bscale = 0.13790548;
float_t k002qlinearmac_zscale = 0.19320115;
float_t k002qlinearmac_yscale = 0.043735545;
uint8_t k002qlinearmac_xzero = 146;
uint8_t k002qlinearmac_wzero = 0;
uint8_t k002qlinearmac_bzero = 150;
uint8_t k002qlinearmac_zzero = 119;
uint8_t k002qlinearmac_yzero = 0;


// instance to be compiled and used in host within xclbin
QlinearMacGraphTest<QlinearMacScalar,uint8_t,uint8_t,14,128,0> qlinearmacScalar(
  "qlinearmacScalar", 
  k002qlinearmac_w, k002qlinearmac_b, k002qlinearmac_xscale, k002qlinearmac_wscale, k002qlinearmac_bscale, k002qlinearmac_zscale, k002qlinearmac_yscale, k002qlinearmac_xzero, k002qlinearmac_wzero, k002qlinearmac_bzero, k002qlinearmac_zzero, k002qlinearmac_yzero,
  "k002qlinearmac_in_shape14x128.txt", 
  "k002qlinearmac_goldenout_shape14x128_scalar.txt");

QlinearMacStreamGraphTest<QlinearMac,uint8_t,uint8_t,14,128,0> qlinearmac(
  "qlinearmac", 
  k002qlinearmac_w, k002qlinearmac_b, k002qlinearmac_xscale, k002qlinearmac_wscale, k002qlinearmac_bscale, k002qlinearmac_zscale, k002qlinearmac_yscale, k002qlinearmac_xzero, k002qlinearmac_wzero, k002qlinearmac_bzero, k002qlinearmac_zzero, k002qlinearmac_yzero,
  "k002qlinearmac_in_shape14x128.txt", 
  "k002qlinearmac_goldenout_shape14x128.txt");


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