#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "gemm.h"


template <int M, int K, int N, 
  const char* INP_TXT, const char* WEIGHT_TXT, const char* BIAS_TXT, const char* OUT_TXT>
class GemmReluScalar : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::input_plio plin[3];
    adf::output_plio plout[1];

    GemmReluScalar(const std::string& id) { 
      this->id = id;

      k[0] = adf::kernel::create(gemm_relu_scalar<M, K, N>);
      adf::source(k[0]) = "gemm.cc";

#ifdef EXTERNAL_IO
      plin[0] = adf::input_plio::create("plin0_gemm"+id+"_input", adf::plio_64_bits);
      plin[1] = adf::input_plio::create("plin1_gemm"+id+"_weight", adf::plio_64_bits);
      plin[2] = adf::input_plio::create("plin2_gemm"+id+"_bias", adf::plio_64_bits);
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", adf::plio_64_bits);
#else
      plin[0] = adf::input_plio::create("plin0_gemm"+id+"_input", adf::plio_64_bits, INP_TXT);
      plin[1] = adf::input_plio::create("plin1_gemm"+id+"_weight", adf::plio_64_bits, WEIGHT_TXT);
      plin[2] = adf::input_plio::create("plin2_gemm"+id+"_bias", adf::plio_64_bits, BIAS_TXT);
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif
      
      adf::connect<adf::window<M*K*4>> (plin[0].out[0], k[0].in[0]);
      adf::connect<adf::window<K*N*4>> (plin[1].out[0], k[0].in[1]);
      adf::connect<adf::window<N*4>> (plin[2].out[0], k[0].in[2]);
      adf::connect<adf::window<M*N*4>> (k[0].out[0], plout[0].in[0]);

      adf::runtime<ratio>(k[0]) = 0.6;
    }

};


#endif // __GRAPH_H__