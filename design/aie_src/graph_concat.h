#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "concat.h"


template <int LCNT, int NCHUNK, int OUTSIZE>
class ConcatScalarGraph : public adf::graph {

  private:
    adf::kernel k[1];
    std::string id;
    static const int NLANES = 32;

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
      const std::string& INP8_TXT,
      const std::string& INP9_TXT,
      const std::string& INP10_TXT,
      const std::string& INP11_TXT,
      const std::string& INP12_TXT,
      const std::string& INP13_TXT,
      const std::string& INP14_TXT,
      const std::string& INP15_TXT,
      const std::string& INP16_TXT,
      const std::string& INP17_TXT,
      const std::string& INP18_TXT,
      const std::string& INP19_TXT,
      const std::string& INP20_TXT,
      const std::string& INP21_TXT,
      const std::string& INP22_TXT,
      const std::string& INP23_TXT,
      const std::string& INP24_TXT,
      const std::string& INP25_TXT,
      const std::string& INP26_TXT,
      const std::string& INP27_TXT,
      const std::string& INP28_TXT,
      const std::string& INP29_TXT,
      const std::string& INP30_TXT,
      const std::string& INP31_TXT,
      const std::string& OUT_TXT = "concat_out.txt"
    ) { 
      this->id = id;

      k[0] = adf::kernel::create(concat32_scalar<NCHUNK, OUTSIZE>);
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
      SET_PLIN(8, INP8_TXT);
      SET_PLIN(9, INP9_TXT);
      SET_PLIN(10, INP10_TXT);
      SET_PLIN(11, INP11_TXT);
      SET_PLIN(12, INP12_TXT);
      SET_PLIN(13, INP13_TXT);
      SET_PLIN(14, INP14_TXT);
      SET_PLIN(15, INP15_TXT);
      SET_PLIN(16, INP16_TXT);
      SET_PLIN(17, INP17_TXT);
      SET_PLIN(18, INP18_TXT);
      SET_PLIN(19, INP19_TXT);
      SET_PLIN(20, INP20_TXT);
      SET_PLIN(21, INP21_TXT);
      SET_PLIN(22, INP22_TXT);
      SET_PLIN(23, INP23_TXT);
      SET_PLIN(24, INP24_TXT);
      SET_PLIN(25, INP25_TXT);
      SET_PLIN(26, INP26_TXT);
      SET_PLIN(27, INP27_TXT);
      SET_PLIN(28, INP28_TXT);
      SET_PLIN(29, INP29_TXT);
      SET_PLIN(30, INP30_TXT);
      SET_PLIN(31, INP31_TXT);
      
      plout[0] = adf::output_plio::create("plout0_concat"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif
      
      adf::connect<adf::window<OUTSIZE*4>> (k[0].out[0], plout[0].in[0]);
    }

};


#endif // __GRAPH_H__