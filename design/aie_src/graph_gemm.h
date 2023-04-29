#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include "gemm.h"
#include "concat.h"


// chunks NxK weight params since they are too large
template <int NCHUNK, int M, int K, int N, 
  const char* INP_TXT, const char* OUT_TXT>
class GemmReluScalarGraph : public adf::graph {

  private:
    static const int CHUNK_COUNT = N / NCHUNK + 1;
    static constexpr int NCUTCHUNK = N % NCHUNK;
    std::vector<adf::kernel> gemms;
    std::string id;

  public:
    std::vector<adf::input_plio> plins;
    std::vector<adf::output_plio> plouts;

    GemmReluScalarGraph(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      this->id = id;

      adf::kernel k;
      adf::input_plio plin;
      adf::output_plio plout;
      std::vector<float> wChunk;
      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*NCHUNK + NCHUNK > N) ? NCUTCHUNK : NCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*NCHUNK*K, weights.begin()+(i*NCHUNK+chunkSize)*K); 
        wChunk.resize(NCHUNK*K, 0);
        bChunk = std::vector<float>(bias.begin()+i*NCHUNK, bias.begin()+i*NCHUNK+chunkSize);
        bChunk.resize(NCHUNK, 0);
        k = adf::kernel::create_object<GemmReluScalar<M, K, NCHUNK>>(wChunk, bChunk, i*NCHUNK);
        adf::source(k) = "gemm.cc";
        adf::runtime<ratio>(k) = 0.6;

#ifdef EXTERNAL_IO
        plin = adf::input_plio::create("plin"+std::to_string(i)+"_gemm"+id+"_input", adf::plio_64_bits);
        plout = adf::output_plio::create("plout"+std::to_string(i)+"_gemm"+id+"_output", adf::plio_64_bits);
#else
        plin = adf::input_plio::create("plin"+std::to_string(i)+"_gemm"+id+"_input", adf::plio_64_bits, INP_TXT);
        plout = adf::output_plio::create("plout"+std::to_string(i)+"_gemm"+id+"_output", adf::plio_64_bits, OUT_TXT+std::to_string(i));
#endif  
        
        adf::connect<adf::window<M*K*4>> (plin.out[0], k.in[0]);
        adf::connect<adf::window<M*NCHUNK*4>> (k.out[0], plout.in[0]);
        gemms.push_back(k);
        plins.push_back(plin);
        plouts.push_back(plout);
      }

    }

};


#endif // __GRAPH_H__