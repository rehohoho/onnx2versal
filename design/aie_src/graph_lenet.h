#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "graph_argmax.h"
#include "graph_conv.h"
#include "graph_gemm.h"
#include "graph_pool.h"
#include "graph_utils.h"


/**
 * @defgroup Lenet
 * 
 * @tparam CONV   Conv2D Kernel
 * @tparam POOL   Pool2D Kernel
 * @tparam GEMM   Gemm Kernel
 * @tparam ARGMAX Argmax Kernel
 * @tparam CONCAT Concat Kernel
 * 
 * @{
 */

/**
 * @brief Lenet on Mnist dataset, assumes BCHW format
 * 
 * @details
 * Profile information
 * - Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6> total = 16665
 * - Running Maxpool2x2BCHW::filter<24, 12, 1, 6> total = 901
 * - Running Conv5x5on8ReluBCHW<12, 8, 1, 6, 16> total = 25952
 * - Running Maxpool2x2BCHW::filter<8, 4, 1, 16> total = 291
 * - Running 8x GemmReluMKKN<1, 256, 16> total = 906
 * - Running ConcatVector<8, 16, 16, 120>::filter8 total = 126
 * - Running 3x GemmReluMKKN<1, 120, 32> total = 888
 * - Running ConcatVector<3, 32, 32, 84>::filter3 total = 73
 * - Running GemmReluMKKN<1, 84, 48> total = 973
 * - Running ConcatVector<1, 48, 48, 10>::filter7 total = 41
 * - Running ArgmaxScalar<10, 10> total = 136
 * 

 * 
 * @connections
 * @connect{pin[1], 1*1*28*28*4}
 * @connect{pout[?], ?}
 * @endconnections
 */
template <
  template<int, int, int, int, int, int> class CONV,
  template<int, int, int, int> class POOL,
  template<int, int, int> class GEMM,
  template<int, int> class ARGMAX,
  template<int, int, int, int> class CONCAT>
class MnistLenetBchwGraph : public adf::graph {

  private:
    typedef GemmReluMkknChunkGraph<GEMM, ConcatVector, MAX_FLOAT_PARAMS/256/4*4, 1, 256, 120> Gemm1;
    typedef GemmReluMkknChunkGraph<GEMM, ConcatVector, MAX_FLOAT_PARAMS/120/4*4, 1, 120, 84> Gemm2;
    typedef GemmReluMkknChunkGraph<GEMM, ConcatVector, MAX_FLOAT_PARAMS/84/4*4, 1, 84, 10> Gemm3;

    ConvReluGraph<CONV, 28, 24, 1, 1, 6, 5> k0conv1;
    MaxpoolGraph<POOL, 24, 12, 1, 6> k1pool1;
    ConvReluGraph<CONV, 12, 8, 1, 6, 16, 5> k2conv2;
    MaxpoolGraph<POOL, 8, 4, 1, 16> k3pool2;
    Gemm1 k4gemm1;
    Gemm2 k5gemm2;
    Gemm3 k6gemm3;
    ArgmaxGraph<ARGMAX, 10, 10> k7argmax1;

  public:
    adf::input_plio plin[1];
    std::vector<adf::output_plio> plout; // intermediate outputs optional

    MnistLenetBchwGraph(
      const std::string& id,
      const std::string& INPUT_TXT,
      const std::string& OUTPUT_TXT,
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
      // plout[0] is mandatory output
      adf::output_plio a = adf::output_plio::create("plout0_"+id+"_argm19", PLIO64_ARG(OUTPUT_TXT));
      plout.push_back(a);
      adf::connect<adf::window<1*10*4>> (k7argmax1.pout[0], a.in[0]);

#define SET_OPT_PLOUT(TXT_PATH, STMT, PLOUT_NAME) \
      if (!TXT_PATH.empty()) { \
        std::string plout_name = "plout"+std::to_string(plout.size())+"_"+id+"_"+PLOUT_NAME; \
        adf::output_plio a = adf::output_plio::create(plout_name, PLIO64_ARG(TXT_PATH)); \
        STMT; plout.push_back(a);} 

      // plout[1] onwards optional output
      SET_OPT_PLOUT(OUT_CONV0, adf::connect<adf::window<1*24*24*6*4>> (k0conv1.pout[0], a.in[0]), "conv00");
      SET_OPT_PLOUT(OUT_POOL1, adf::connect<adf::window<1*12*12*6*4>> (k1pool1.pout[0], a.in[0]), "pool01");
      SET_OPT_PLOUT(OUT_CONV2, adf::connect<adf::window<1*8*8*16*4>>  (k2conv2.pout[0], a.in[0]), "conv02");
      SET_OPT_PLOUT(OUT_POOL3, adf::connect<adf::window<1*4*4*16*4>>  (k3pool2.pout[0], a.in[0]), "pool03");
      SET_OPT_PLOUT(OUT_GEMM4, adf::connect<adf::window<1*120*4>>     (k4gemm1.pout[0], a.in[0]), "gemm14");
      SET_OPT_PLOUT(OUT_GEMM5, adf::connect<adf::window<1*84*4>>      (k5gemm2.pout[0], a.in[0]), "gemm16");
      SET_OPT_PLOUT(OUT_GEMM6, adf::connect<adf::window<1*10*4>>      (k6gemm3.pout[0], a.in[0]), "gemm18");

      // plin[0] mandatory input
      plin[0] = adf::input_plio::create("plin0_"+id+"_input", PLIO64_ARG(INPUT_TXT));
      adf::connect<adf::window<1*28*28*1*4>> (plin[0].out[0], k0conv1.pin[0]);
      
      // interkernel
      adf::connect<adf::window<1*24*24*6*4>>  (k0conv1.pout[0], k1pool1.pin[0]);
      adf::connect<adf::window<1*12*12*6*4>>  (k1pool1.pout[0], k2conv2.pin[0]);
      adf::connect<adf::window<1*8*8*16*4>>   (k2conv2.pout[0], k3pool2.pin[0]);

      for (int i = 0; i < Gemm1::CHUNK_COUNT; i++)
        adf::connect<adf::window<1*256*4>> (k3pool2.pout[0], k4gemm1.pin[i]);
      for (int i = 0; i < Gemm2::CHUNK_COUNT; i++)
        adf::connect<adf::window<1*120*4>> (k4gemm1.pout[0], k5gemm2.pin[i]);
      for (int i = 0; i < Gemm3::CHUNK_COUNT; i++)
        adf::connect<adf::window<1*84*4>> (k5gemm2.pout[0], k6gemm3.pin[i]);
      
      adf::connect<adf::window<1*10*4>> (k6gemm3.pout[0], k7argmax1.pin[0]);
    }

};
/** @} */


#endif // __GRAPH_H__