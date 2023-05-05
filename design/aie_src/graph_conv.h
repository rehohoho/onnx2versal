#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "conv.h"
#include "concat.h"


template <template<int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluTemplateGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    void init(
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      k[0] = adf::kernel::create_object<CONV<INP_W, OUT_W, B, C, M, K>>(weights, bias);
      adf::source(k[0]) = "conv.cc";

      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[0], k[0].in[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (k[0].out[0], pout[0]);
      adf::runtime<ratio>(k[0]) = 0.6;
    }

};


#define CHUNK_COUNT   M / MCHUNK + 1
#define MCUTCHUNK     M % MCHUNK
#define CONCAT_NLANES 8

template <int MCHUNK, int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluScalarBhwcChunkGraph : public adf::graph {

  private:
    adf::kernel convs[CHUNK_COUNT];
    adf::kernel concat;
    std::string id;

  public:
    adf::input_plio plins[CONCAT_NLANES];
    adf::output_plio plouts[1];
    // adf::output_plio plouts[CONCAT_NLANES];

    ConvReluScalarBhwcChunkGraph(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& EMPTY_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ) { 
      this->id = id;

      concat = adf::kernel::create(
        concat8_scalar<CHUNK_COUNT, B*OUT_W*OUT_W*MCHUNK, MCHUNK, M>);
      adf::source(concat) = "concat.cc";
      adf::runtime<ratio>(concat) = 0.6;

      std::vector<float> wChunk;
      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*MCHUNK + MCHUNK > M) ? MCUTCHUNK : MCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*MCHUNK*K*K*C, weights.begin()+(i*MCHUNK+chunkSize)*K*K*C); 
        wChunk.resize(MCHUNK*K*K*C, 0);
        bChunk = std::vector<float>(bias.begin()+i*MCHUNK, bias.begin()+i*MCHUNK+chunkSize);
        bChunk.resize(MCHUNK, 0);

        convs[i] = adf::kernel::create_object<ConvReluScalarBHWC<INP_W, OUT_W, B, C, MCHUNK, K>>(
          wChunk, bChunk);
        adf::source(convs[i]) = "conv.cc";
        adf::runtime<ratio>(convs[i]) = 0.6;
      }

#ifdef EXTERNAL_IO
#define SET_PLIN(i, TXT_PATH) { \
      std::string plio_name = "plin"+std::to_string(i)+"_conv"+id+"_input"; \
      plins[i] = adf::input_plio::create(plio_name, adf::plio_64_bits); }

      plouts[0] = adf::output_plio::create("plout0_conv"+id+"_output", adf::plio_64_bits);
#else
#define SET_PLIN(i, TXT_PATH) { \
      std::string plio_name = "plin"+std::to_string(i)+"_conv"+id+"_input"; \
      plins[i] = adf::input_plio::create(plio_name, adf::plio_64_bits, TXT_PATH); }
      
      plouts[0] = adf::output_plio::create("plout0_conv"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif

      for (int i = 0; i < CONCAT_NLANES; i++) {
        if (i < CHUNK_COUNT) {
          SET_PLIN(i, INP_TXT);
          adf::connect<adf::window<B*INP_W*INP_W*C*4>> (plins[i].out[0], convs[i].in[0]);
          adf::connect<adf::window<B*OUT_W*OUT_W*MCHUNK*4>> (convs[i].out[0], concat.in[i]);
          // plouts[i] = adf::output_plio::create("plout"+std::to_string(i)+"_conv"+id+"_output", adf::plio_64_bits, OUT_TXT+std::to_string(i));
          // adf::connect<adf::window<B*OUT_W*OUT_W*MCHUNK*4>> (convs[i].out[0], plouts[i].in[0]);
        } else {
          SET_PLIN(i, EMPTY_TXT);
          adf::connect<adf::window<4>> (plins[i].out[0], concat.in[i]);
        }
      }
      
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (concat.out[0], plouts[0].in[0]);
    }

};


#endif // __GRAPH_H__