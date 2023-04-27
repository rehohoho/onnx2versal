#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "conv.h"
#include "pool.h"
#include "transpose.h"
#include "gemm.h"


template <
  const char* INPUT_TXT, 
  const char* CONV01_W_TXT, 
  const char* CONV01_B_TXT,
  const char* CONV03_W_TXT, 
  const char* CONV03_B_TXT,
  const char* GEMM14_W_TXT,
  const char* GEMM14_B_TXT,
  const char* GEMM16_W_TXT,
  const char* GEMM16_B_TXT,
  const char* GEMM18_W_TXT,
  const char* GEMM18_B_TXT,
  const char* OUT_CONV01,
  const char* OUT_POOL02,
  const char* OUT_CONV03,
  const char* OUT_POOL05,
  const char* OUT_TRAN05,
  const char* OUT_GEMM14,
  const char* OUT_GEMM16,
  const char* OUT_GEMM18
>
class MnistLenetScalar : public adf::graph {

  private:
    adf::kernel k[8];
    std::string id;

  public:
    adf::input_plio plin[11];
    adf::output_plio plout[8];

    MnistLenetScalar(const std::string& id) { 
      this->id = id;

      k[0] = adf::kernel::create(conv_relu_scalar<28, 24, 1, 1, 6, 5>); adf::source(k[0]) = "conv.cc";
      k[1] = adf::kernel::create(maxpool_scalar<24, 12, 1, 6>); adf::source(k[1]) = "pool.cc";
      k[2] = adf::kernel::create(conv_relu_scalar<12, 8, 1, 6, 16, 5>); adf::source(k[2]) = "conv.cc";
      k[3] = adf::kernel::create(maxpool_scalar<8, 4, 1, 16>); adf::source(k[3]) = "pool.cc";
      k[4] = adf::kernel::create(bhwc2bchw_scalar<1, 4, 4, 16>); adf::source(k[4]) = "transpose.cc";
      k[5] = adf::kernel::create(gemm_relu_scalar<1, 256, 120>); adf::source(k[5]) = "gemm.cc";
      k[6] = adf::kernel::create(gemm_relu_scalar<1, 120, 84>); adf::source(k[6]) = "gemm.cc";
      k[7] = adf::kernel::create(gemm_relu_scalar<1, 84, 10>); adf::source(k[7]) = "gemm.cc";

#ifdef EXTERNAL_IO
      plin[0] = adf::input_plio::create("plin00_"+id+"_inp", adf::plio_64_bits);
      plin[1] = adf::input_plio::create("plin01_"+id+"_conv00w", adf::plio_64_bits);
      plin[2] = adf::input_plio::create("plin02_"+id+"_conv00b", adf::plio_64_bits);
      plin[3] = adf::input_plio::create("plin03_"+id+"_conv03w", adf::plio_64_bits);
      plin[4] = adf::input_plio::create("plin04_"+id+"_conv03b", adf::plio_64_bits);
      plin[5] = adf::input_plio::create("plin05_"+id+"_gemm14w", adf::plio_64_bits);
      plin[6] = adf::input_plio::create("plin06_"+id+"_gemm14b", adf::plio_64_bits);
      plin[7] = adf::input_plio::create("plin07_"+id+"_gemm16w", adf::plio_64_bits);
      plin[8] = adf::input_plio::create("plin08_"+id+"_gemm16b", adf::plio_64_bits);
      plin[9] = adf::input_plio::create("plin09_"+id+"_gemm18w", adf::plio_64_bits);
      plin[10] = adf::input_plio::create("plin10_"+id+"_gemm18b", adf::plio_64_bits);
      plout[0] = adf::output_plio::create("plout0_"+id+"_conv00", adf::plio_64_bits);
      plout[1] = adf::output_plio::create("plout1_"+id+"_pool02", adf::plio_64_bits);
      plout[2] = adf::output_plio::create("plout2_"+id+"_conv03", adf::plio_64_bits);
      plout[3] = adf::output_plio::create("plout3_"+id+"_pool05", adf::plio_64_bits);
      plout[4] = adf::output_plio::create("plout4_"+id+"_tran05", adf::plio_64_bits);
      plout[5] = adf::output_plio::create("plout5_"+id+"_gemm14", adf::plio_64_bits);
      plout[6] = adf::output_plio::create("plout6_"+id+"_gemm16", adf::plio_64_bits);
      plout[7] = adf::output_plio::create("plout7_"+id+"_gemm18", adf::plio_64_bits);
#else
      plin[0] = adf::input_plio::create("plin00_"+id+"_inp", adf::plio_64_bits, INPUT_TXT);
      plin[1] = adf::input_plio::create("plin01_"+id+"_conv00w", adf::plio_64_bits, CONV01_W_TXT);
      plin[2] = adf::input_plio::create("plin02_"+id+"_conv00b", adf::plio_64_bits, CONV01_B_TXT);
      plin[3] = adf::input_plio::create("plin03_"+id+"_conv03w", adf::plio_64_bits, CONV03_W_TXT);
      plin[4] = adf::input_plio::create("plin04_"+id+"_conv03b", adf::plio_64_bits, CONV03_B_TXT);
      plin[5] = adf::input_plio::create("plin05_"+id+"_gemm14w", adf::plio_64_bits, GEMM14_W_TXT);
      plin[6] = adf::input_plio::create("plin06_"+id+"_gemm14b", adf::plio_64_bits, GEMM14_B_TXT);
      plin[7] = adf::input_plio::create("plin07_"+id+"_gemm16w", adf::plio_64_bits, GEMM16_W_TXT);
      plin[8] = adf::input_plio::create("plin08_"+id+"_gemm16b", adf::plio_64_bits, GEMM16_B_TXT);
      plin[9] = adf::input_plio::create("plin09_"+id+"_gemm18w", adf::plio_64_bits, GEMM18_W_TXT);
      plin[10] = adf::input_plio::create("plin10_"+id+"_gemm18b", adf::plio_64_bits, GEMM18_B_TXT);
      plout[0] = adf::output_plio::create("plout0_"+id+"_conv00", adf::plio_64_bits, OUT_CONV01);
      plout[1] = adf::output_plio::create("plout1_"+id+"_pool02", adf::plio_64_bits, OUT_POOL02);
      plout[2] = adf::output_plio::create("plout2_"+id+"_conv03", adf::plio_64_bits, OUT_CONV03);
      plout[3] = adf::output_plio::create("plout3_"+id+"_pool05", adf::plio_64_bits, OUT_POOL05);
      plout[4] = adf::output_plio::create("plout4_"+id+"_tran05", adf::plio_64_bits, OUT_TRAN05);
      plout[5] = adf::output_plio::create("plout5_"+id+"_gemm14", adf::plio_64_bits, OUT_GEMM14);
      plout[6] = adf::output_plio::create("plout6_"+id+"_gemm16", adf::plio_64_bits, OUT_GEMM16);
      plout[7] = adf::output_plio::create("plout7_"+id+"_gemm18", adf::plio_64_bits, OUT_GEMM18);
#endif
      
      // interkernel
      adf::connect<adf::window<1*24*24*6*4>>  (k[0].out[0], k[1].in[0]); // pool02 in
      adf::connect<adf::window<1*12*12*6*4>>  (k[1].out[0], k[2].in[0]); // conv03 in
      adf::connect<adf::window<1*8*8*16*4>>   (k[2].out[0], k[3].in[0]); // pool05 in
      adf::connect<adf::window<1*4*4*16*4>>   (k[3].out[0], k[4].in[0]); // tran05 in
      adf::connect<adf::window<1*256*4>>      (k[4].out[0], k[5].in[0]); // gemm14 in
      adf::connect<adf::window<1*120*4>>      (k[5].out[0], k[6].in[0]); // gemm16 in
      adf::connect<adf::window<1*84*4>>       (k[6].out[0], k[7].in[0]); // gemm18 in
      
      // outputs
      adf::connect<adf::window<1*24*24*6*4>>  (k[0].out[0], plout[0].in[0]); // conv00
      adf::connect<adf::window<1*12*12*6*4>>  (k[1].out[0], plout[1].in[0]); // pool02
      adf::connect<adf::window<1*8*8*16*4>>   (k[2].out[0], plout[2].in[0]); // conv03
      adf::connect<adf::window<1*4*4*16*4>>   (k[3].out[0], plout[3].in[0]); // pool05
      adf::connect<adf::window<1*256*4>>      (k[4].out[0], plout[4].in[0]); // tran05
      adf::connect<adf::window<1*120*4>>      (k[5].out[0], plout[5].in[0]); // gemm14
      adf::connect<adf::window<1*84*4>>       (k[6].out[0], plout[6].in[0]); // gemm16
      adf::connect<adf::window<1*10*4>>       (k[7].out[0], plout[7].in[0]); // gemm18
      
      // inputs and parameters
      adf::connect<adf::window<1*28*28*1*4>> (plin[0].out[0], k[0].in[0]);// input
      adf::connect<adf::window<6*5*5*1*4>>  (plin[1].out[0], k[0].in[1]); // conv00
      adf::connect<adf::window<6*4>>        (plin[2].out[0], k[0].in[2]);
      adf::connect<adf::window<16*5*5*6*4>> (plin[3].out[0], k[2].in[1]); // conv03
      adf::connect<adf::window<16*4>>       (plin[4].out[0], k[2].in[2]);
      
      adf::connect<adf::window<120*256*4>>  (plin[5].out[0], k[5].in[1]); // gemm14
      adf::connect<adf::window<120*4>>      (plin[6].out[0], k[5].in[2]);
      adf::connect<adf::window<84*120*4>>   (plin[7].out[0], k[6].in[1]); // gemm16
      adf::connect<adf::window<84*4>>       (plin[8].out[0], k[6].in[2]);
      adf::connect<adf::window<10*84*4>>    (plin[9].out[0], k[7].in[1]); // gemm18
      adf::connect<adf::window<10*4>>       (plin[10].out[0], k[7].in[2]);
      
      for (int i = 0; i < 8; i++)
        adf::runtime<ratio>(k[i]) = 0.6;
    }

};


#endif // __GRAPH_H__