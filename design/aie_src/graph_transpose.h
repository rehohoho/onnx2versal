#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "transpose.h"


template <int B, int H, int W, int C, const char* INP_TXT, const char* OUT_TXT>
class TransposeScalar : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    TransposeScalar(const std::string& id) { 
      this->id = id;

      k[0] = adf::kernel::create(bhwc2bchw_scalar<B, H, W, C>);
      adf::source(k[0]) = "transpose.cc";

#ifdef EXTERNAL_IO
      plin[0] = adf::input_plio::create("plin0_transpose"+id+"_input", adf::plio_64_bits);
      plout[0] = adf::output_plio::create("plout0_transpose"+id+"_output", adf::plio_64_bits);
#else
      plin[0] = adf::input_plio::create("plin0_transpose"+id+"_input", adf::plio_64_bits, INP_TXT);
      plout[0] = adf::output_plio::create("plout0_transpose"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif
      
      adf::connect<adf::window<B*H*W*C*4>> (plin[0].out[0], k[0].in[0]);
      adf::connect<adf::window<B*C*H*W*4>> (k[0].out[0], plout[0].in[0]);

      adf::runtime<ratio>(k[0]) = 0.6;
    }

};


#endif // __GRAPH_H__