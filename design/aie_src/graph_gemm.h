#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "gemm.h"
#include "concat.h"


// chunks NxK weight params since they are too large
template <int NCHUNK, int M, int K, int N>
class GemmReluScalarGraph : public adf::graph {

  private:
    static const int CHUNK_COUNT = N / NCHUNK + 1;
    static constexpr int NCUTCHUNK = N % NCHUNK;
    adf::kernel gemms[CHUNK_COUNT];
    adf::kernel concat;
    std::string id;

  public:
    adf::input_plio plins[CHUNK_COUNT];
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

      concat = adf::kernel::create(concat4_scalar<NCHUNK, M*N>);
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
        gemms[i] = adf::kernel::create_object<GemmReluScalar<M, K, NCHUNK>>(wChunk, bChunk, i*NCHUNK);
        adf::source(gemms[i]) = "gemm.cc";
        adf::runtime<ratio>(gemms[i]) = 0.6;

#ifdef EXTERNAL_IO
        plins[i] = adf::input_plio::create("plin"+std::to_string(i)+"_gemm"+id+"_input", adf::plio_64_bits);
#else
        plins[i] = adf::input_plio::create("plin"+std::to_string(i)+"_gemm"+id+"_input", adf::plio_64_bits, INP_TXT);
#endif
        
        adf::connect<adf::window<M*K*4>> (plins[i].out[0], gemms[i].in[0]);
        adf::connect<adf::window<M*NCHUNK*4>> (gemms[i].out[0], concat.in[i]);
      }

      for (int i = CHUNK_COUNT; i < 4; i++) {
#ifdef EXTERNAL_IO
        plins[i] = adf::input_plio::create("plin"+std::to_string(i)+"_gemm"+id+"_input", adf::plio_64_bits);
#else
        plins[i] = adf::input_plio::create("plin"+std::to_string(i)+"_gemm"+id+"_input", adf::plio_64_bits, EMPTY_TXT);
#endif
        adf::connect<adf::window<4>> (plins[i].out[0], concat.in[i]);
      }

#ifdef EXTERNAL_IO
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", adf::plio_64_bits);
#else
      plout[0] = adf::output_plio::create("plout0_gemm"+id+"_output", adf::plio_64_bits, OUT_TXT);
#endif  
      adf::connect<adf::window<M*N*4>> (concat.out[0], plout[0].in[0]);
    }

};


#endif // __GRAPH_H__