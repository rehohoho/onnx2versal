#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "concat.h"


template <int LCNT, int NCHUNK, int OUTSIZE>
class ConcatScalarGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;
    static const int NLANES = 8;

  public:
    adf::input_plio plin[NLANES];
    adf::output_plio plout[1];

    ConcatScalarGraph(
      const std::string& id,
      const std::string& INP0_TXT,
      const std::string& INP1_TXT,
      const std::string& INP2_TXT,
      const std::string& INP3_TXT,
      const std::string& INP4_TXT,
      const std::string& INP5_TXT,
      const std::string& INP6_TXT,
      const std::string& INP7_TXT,
      const std::string& OUT_TXT = "concat_out.txt"
    ) { 
      this->id = id;

      k[0] = adf::kernel::create(concat8_scalar<NCHUNK, OUTSIZE>);
      adf::source(k[0]) = "concat.cc";
      adf::runtime<ratio>(k[0]) = 0.6;

#ifdef EXTERNAL_IO
      for (int i = 0; i < NLANES; i++) {
        std::string plin_name = "plin"+std::to_string(i)+"_concat"+id+"_input";
        plin[i] = adf::input_plio::create(plin_name, adf::plio_64_bits);
        if (LCNT > i) {
          adf::connect<adf::window<NCHUNK*4>> (plin[i].out[0], k[0].in[i]);
        } else {
          adf::connect<adf::window<4>> (plin[i].out[0], k[0].in[i]);
        }
      }
      plout[0] = adf::output_plio::create("plout0_concat"+id+"_output", adf::plio_64_bits);
#else
#define SET_PLIN(i, TXT_PATH) { \
      std::string plin_name = "plin"+std::to_string(i)+"_concat"+id+"_input"; \
      plin[i] = adf::input_plio::create(plin_name, adf::plio_64_bits, TXT_PATH); \
      if (LCNT > i) { \
        adf::connect<adf::window<NCHUNK*4>> (plin[i].out[0], k[0].in[i]); \
      } else { \
        adf::connect<adf::window<4>> (plin[i].out[0], k[0].in[i]); }}
      
      SET_PLIN(0, INP0_TXT);
      SET_PLIN(1, INP1_TXT);
      SET_PLIN(2, INP2_TXT);
      SET_PLIN(3, INP3_TXT);
      SET_PLIN(4, INP4_TXT);
      SET_PLIN(5, INP5_TXT);
      SET_PLIN(6, INP6_TXT);
      SET_PLIN(7, INP7_TXT);
      
      plout[0] = adf::output_plio::create("plout0_concat"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif
      
      adf::connect<adf::window<OUTSIZE*4>> (k[0].out[0], plout[0].in[0]);
    }

};


#endif // __GRAPH_H__