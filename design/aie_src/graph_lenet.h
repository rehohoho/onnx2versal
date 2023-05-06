#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "graph_conv.h"
#include "graph_gemm.h"
#include "graph_pool.h"
#include "graph_utils.h"


/*
Profile cycles:
Running Conv5x5ReluBCHW<28, 24, 1, 1, 6>: 21325
Running MaxpoolScalarBCHW::filter<24, 12, 1, 6>: 11358
Running Conv5x5ReluBCHW<12, 8, 1, 6, 16>: 41580
Running MaxpoolScalarBCHW::filter<8, 4, 1, 16>: 3490
Running 8x gemm_relu_scalar<1, 256, 16>: 8593
Running concat8_scalar<8, 16, 16, 120>: 630
Running 3x gemm_relu_scalar<1, 120, 34>: 8922
Running concat8_scalar<3, 34, 34, 84>: 883
Running 1x gemm_relu_scalar<1, 84, 48>: 9103
Running concat8_scalar<1, 48, 48, 10>: 937
*/
template <
  template<int, int, int, int, int, int> class CONV,
  template<int, int, int, int> class POOL,
  template<int, int, int> class GEMM
>
class MnistLenetBhwcGraph : public adf::graph {

  private:
    typedef GemmReluChunkGraph<GEMM, MAX_FLOAT_PARAMS/256, 1, 256, 120> Gemm1;
    typedef GemmReluChunkGraph<GEMM, MAX_FLOAT_PARAMS/120, 1, 120, 84> Gemm2;
    typedef GemmReluChunkGraph<GEMM, MAX_FLOAT_PARAMS/84, 1, 84, 10> Gemm3;

    ConvReluGraph<CONV, 28, 24, 1, 1, 6, 5> k0conv1;
    MaxpoolGraph<POOL, 24, 12, 1, 6> k1pool1;
    ConvReluGraph<CONV, 12, 8, 1, 6, 16, 5> k2conv2;
    MaxpoolGraph<POOL, 8, 4, 1, 16> k3pool2;
    Gemm1 k4gemm1;
    Gemm2 k5gemm2;
    Gemm3 k6gemm3;

  public:
    std::vector<adf::input_plio> plin;   // variable empty inputs for chunkers
    std::vector<adf::output_plio> plout; // intermediate outputs optional

    MnistLenetBhwcGraph(
      const std::string& id,
      const std::string& INPUT_TXT,
      const std::string& EMPTY_TXT,
      std::vector<float> conv01w,
      std::vector<float> conv01b,
      std::vector<float> conv02w,
      std::vector<float> conv02b,
      std::vector<float> gemm14w,
      std::vector<float> gemm14b,
      std::vector<float> gemm16w,
      std::vector<float> gemm16b,
      std::vector<float> gemm18w,
      std::vector<float> gemm18b,
      const std::string& OUT_CONV0 = std::string(),
      const std::string& OUT_POOL1 = std::string(),
      const std::string& OUT_CONV2 = std::string(),
      const std::string& OUT_POOL3 = std::string(),
      const std::string& OUT_GEMM4 = std::string(),
      const std::string& OUT_GEMM5 = std::string(),
      const std::string& OUT_GEMM6 = std::string()
    ): 
      k0conv1(conv01w, conv01b), 
      k2conv2(conv02w, conv02b),
      k4gemm1(gemm14w, gemm14b),
      k5gemm2(gemm16w, gemm16b),
      k6gemm3(gemm18w, gemm18b)
    { 

#define SET_OPT_PLOUT(TXT_PATH, STMT, PLOUT_NAME) \
  if (!TXT_PATH.empty()) { \
    adf::output_plio a = adf::output_plio::create( \
      "plout"+std::to_string(plout.size())+"_"+id+"_"+PLOUT_NAME, adf::plio_64_bits, TXT_ARG(TXT_PATH)); \
    STMT; plout.push_back(a);} 

      // optional output
      SET_OPT_PLOUT(OUT_CONV0, adf::connect<adf::window<1*24*24*6*4>> (k0conv1.pout[0], a.in[0]), "conv00");
      SET_OPT_PLOUT(OUT_POOL1, adf::connect<adf::window<1*12*12*6*4>> (k1pool1.pout[0], a.in[0]), "pool01");
      SET_OPT_PLOUT(OUT_CONV2, adf::connect<adf::window<1*8*8*16*4>>  (k2conv2.pout[0], a.in[0]), "conv02");
      SET_OPT_PLOUT(OUT_POOL3, adf::connect<adf::window<1*4*4*16*4>>  (k3pool2.pout[0], a.in[0]), "pool03");
      SET_OPT_PLOUT(OUT_GEMM4, adf::connect<adf::window<1*120*4>>     (k4gemm1.pout[0], a.in[0]), "gemm14");
      SET_OPT_PLOUT(OUT_GEMM5, adf::connect<adf::window<1*84*4>>      (k5gemm2.pout[0], a.in[0]), "gemm16");
      SET_OPT_PLOUT(OUT_GEMM6, adf::connect<adf::window<1*10*4>>      (k6gemm3.pout[0], a.in[0]), "gemm18");

      // input
      adf::input_plio _plin;
      _plin = adf::input_plio::create("plin0_lenet"+id+"_input", adf::plio_64_bits, TXT_ARG(INPUT_TXT));
      adf::connect<adf::window<1*28*28*1*4>> (_plin.out[0], k0conv1.pin[0]);
      plin.push_back(_plin);
      
      // interkernel
      adf::connect<adf::window<1*24*24*6*4>>  (k0conv1.pout[0], k1pool1.pin[0]);
      adf::connect<adf::window<1*12*12*6*4>>  (k1pool1.pout[0], k2conv2.pin[0]);
      adf::connect<adf::window<1*8*8*16*4>>   (k2conv2.pout[0], k3pool2.pin[0]);

      for (int i = 0; i < Gemm1::CONCAT_NLANES; i++) {
        if (i < Gemm1::CHUNK_COUNT) {
          adf::connect<adf::window<1*256*4>> (k3pool2.pout[0], k4gemm1.pin[i]);
        } else {
          std::string plio_name = "plin"+std::to_string(plin.size())+"_lenet"+id+"_input";
          printf("%s ", plio_name.c_str());
          _plin = adf::input_plio::create(plio_name, adf::plio_64_bits, TXT_ARG(EMPTY_TXT));
          adf::connect<adf::window<4>> (_plin.out[0], k4gemm1.pin[i]);
          plin.push_back(_plin);
        }
      }

      for (int i = 0; i < Gemm2::CONCAT_NLANES; i++) {
        if (i < Gemm2::CHUNK_COUNT) {
          adf::connect<adf::window<1*120*4>> (k4gemm1.pout[0], k5gemm2.pin[i]);
        } else {
          std::string plio_name = "plin"+std::to_string(plin.size())+"_lenet"+id+"_input";
          printf("%s ", plio_name.c_str());
          _plin = adf::input_plio::create(plio_name, adf::plio_64_bits, TXT_ARG(EMPTY_TXT));
          adf::connect<adf::window<4>> (_plin.out[0], k5gemm2.pin[i]);
          plin.push_back(_plin);
        }
      }

      for (int i = 0; i < Gemm3::CONCAT_NLANES; i++) {
        if (i < Gemm3::CHUNK_COUNT) {
          adf::connect<adf::window<1*84*4>> (k5gemm2.pout[0], k6gemm3.pin[i]);
        } else {
          std::string plio_name = "plin"+std::to_string(plin.size())+"_lenet"+id+"_input";
          printf("%s ", plio_name.c_str());
          _plin = adf::input_plio::create(plio_name, adf::plio_64_bits, TXT_ARG(EMPTY_TXT));
          adf::connect<adf::window<4>> (_plin.out[0], k6gemm3.pin[i]);
          plin.push_back(_plin);
        }
      }

    }

};


#endif // __GRAPH_H__