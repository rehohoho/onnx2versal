#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "concat.h"


template <int LCNT, int L0, int L1, int L2, int L3, int OUTSIZE>
class ConcatScalarGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;

  public:
    adf::input_plio plin[4];
    adf::output_plio plout[1];

    ConcatScalarGraph(
      const std::string& id,
      const std::string& INP0_TXT = std::string(),
      const std::string& INP1_TXT = std::string(),
      const std::string& INP2_TXT = std::string(),
      const std::string& INP3_TXT = std::string(),
      const std::string& OUT_TXT = "concat_out.txt"
    ) { 
      this->id = id;

      k[0] = adf::kernel::create(concat_scalar<L0, L1, L2, L3, OUTSIZE>);
      adf::source(k[0]) = "concat.cc";
      adf::runtime<ratio>(k[0]) = 0.6;

#ifdef EXTERNAL_IO
#define SET_PLIN(i, LSIZE) { \
      std::string plin_name = "plin"+std::to_string(i)+"_concat"+id+"_input"; \
      plin[i] = adf::input_plio::create(plin_name, adf::plio_64_bits); \
      if (LCNT > i) { \
        adf::connect<adf::window<LSIZE*4>> (plin[i].out[0], k[0].in[i]); \
      } else { \
        adf::connect<adf::window<4>> (plin[i].out[0], k[0].in[i]); }}

      SET_PLIN(0, L0);
      SET_PLIN(1, L1);
      SET_PLIN(2, L2);
      SET_PLIN(3, L3);
      plout[0] = adf::output_plio::create("plout0_concat"+id+"_output", adf::plio_64_bits);
#else
#define SET_PLIN(i, LSIZE, TXT_PATH) { \
      std::string plin_name = "plin"+std::to_string(i)+"_concat"+id+"_input"; \
      plin[i] = adf::input_plio::create(plin_name, adf::plio_64_bits, TXT_PATH); \
      if (LCNT > i) { \
        adf::connect<adf::window<LSIZE*4>> (plin[i].out[0], k[0].in[i]); \
      } else { \
        adf::connect<adf::window<4>> (plin[i].out[0], k[0].in[i]); }}
      
      SET_PLIN(0, L0, INP0_TXT);
      SET_PLIN(1, L1, INP1_TXT);
      SET_PLIN(2, L2, INP2_TXT);
      SET_PLIN(3, L3, INP3_TXT);
      plout[0] = adf::output_plio::create("plout0_concat"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif
      
      adf::connect<adf::window<OUTSIZE*4>> (k[0].out[0], plout[0].in[0]);
    }

};


#endif // __GRAPH_H__