#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "gemm.h"
#include "concat.h"


// loads weight into heap
// chunks NxK weight params since they are too large
template <int NCHUNK, int M, int K, int N>
class GemmReluScalarGraph : public adf::graph {

  private:
    static const int CONCAT_NLANES = 8;
    static const int CHUNK_COUNT = N / NCHUNK + 1;
    static constexpr int NCUTCHUNK = N % NCHUNK;
    adf::kernel gemms[CHUNK_COUNT];
    adf::kernel concat;
    std::string id;

  public:
    adf::input_plio plins[CONCAT_NLANES];
    adf::output_plio plout[1];

    GemmReluScalarGraph(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& EMPTY_TXT,
      const std::string& OUT_TXT = "gemm_out.txt"
    ) { 
      this->id = id;

      concat = adf::kernel::create(concat8_scalar<NCHUNK, M*N>);
      adf::source(concat) = "concat.cc";
      adf::runtime<ratio>(concat) = 0.6;

      std::vector<float> wChunk;
      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*NCHUNK + NCHUNK > N) ? NCUTCHUNK : NCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*NCHUNK*K, weights.begin()+(i*NCHUNK+chunkSize)*K); 
        wChunk.resize(NCHUNK*K, 0);
        bChunk = std::vector<float>(bias.begin()+i*NCHUNK, bias.begin()+i*NCHUNK+chunkSize);
        bChunk.resize(NCHUNK, 0);
        gemms[i] = adf::kernel::create_object<GemmReluScalar<M, K, NCHUNK>>(wChunk, bChunk);
        adf::source(gemms[i]) = "gemm.cc";
        adf::runtime<ratio>(gemms[i]) = 0.6;
      }

#ifdef EXTERNAL_IO
#define SET_PLIN(i, TXT_PATH) { \
      std::string plio_name = "plin"+std::to_string(i)+"_gemm"+id+"_input"; \
      plins[i] = adf::input_plio::create(plio_name, adf::plio_64_bits); }
      
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", adf::plio_64_bits);
#else
#define SET_PLIN(i, TXT_PATH) { \
      std::string plio_name = "plin"+std::to_string(i)+"_gemm"+id+"_input"; \
      plins[i] = adf::input_plio::create(plio_name, adf::plio_64_bits, TXT_PATH); }

      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif

      for (int i = 0; i < CONCAT_NLANES; i++) {
        if (i < CHUNK_COUNT) {
          SET_PLIN(i, INP_TXT);
          adf::connect<adf::window<M*K*4>> (plins[i].out[0], gemms[i].in[0]);
          adf::connect<adf::window<M*NCHUNK*4>> (gemms[i].out[0], concat.in[i]);
        } else {
          SET_PLIN(i, EMPTY_TXT);
          adf::connect<adf::window<4>> (plins[i].out[0], concat.in[i]);
        }
      }

      adf::connect<adf::window<M*N*4>> (concat.out[0], plout[0].in[0]);
    }

};


#endif // __GRAPH_H__