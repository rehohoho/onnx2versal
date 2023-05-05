#ifndef __CONV_GRAPH_H__
#define __CONV_GRAPH_H__

#include <adf.h>
#include "conv.h"
#include "concat.h"


template <template<int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    void construct(
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

template <template<int, int, int, int, int, int> class CONV, int IS_BCHW,
  int MCHUNK, int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluChunkGraph : public adf::graph {

  private:
    adf::kernel convs[CHUNK_COUNT];
    adf::kernel concat;

  public:
    adf::port<input> pin[CONCAT_NLANES];
    adf::port<output> pout[1];

    void construct(
      std::vector<float> weights,
      std::vector<float> bias
    ) { 
      if (IS_BCHW) {
        concat = adf::kernel::create_object<ConcatScalar<CHUNK_COUNT, B*MCHUNK*OUT_W*OUT_W, MCHUNK*OUT_W*OUT_W, M*OUT_W*OUT_W>>();
      } else {
        concat = adf::kernel::create_object<ConcatScalar<CHUNK_COUNT, B*OUT_W*OUT_W*MCHUNK, MCHUNK, M>>();
      }
      adf::source(concat) = "concat.cc";
      adf::runtime<ratio>(concat) = 0.6;

      std::vector<float> wChunk;
      std::vector<float> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        int chunkSize = (i*MCHUNK + MCHUNK > M) ? MCUTCHUNK : MCHUNK;
        wChunk = std::vector<float>(weights.begin()+i*MCHUNK*K*K*C, 
                                    weights.begin()+(i*MCHUNK+chunkSize)*K*K*C); 
        wChunk.resize(MCHUNK*K*K*C, 0);
        bChunk = std::vector<float>(bias.begin()+i*MCHUNK, bias.begin()+i*MCHUNK+chunkSize);
        bChunk.resize(MCHUNK, 0);

        convs[i] = adf::kernel::create_object<CONV<INP_W, OUT_W, B, C, MCHUNK, K>>(
          wChunk, bChunk);
        adf::source(convs[i]) = "conv.cc";
        adf::runtime<ratio>(convs[i]) = 0.6;
      }

      for (int i = 0; i < CONCAT_NLANES; i++) {
        if (i < CHUNK_COUNT) {
          adf::connect<adf::window<B*INP_W*INP_W*C*4>> (pin[i], convs[i].in[0]);
          adf::connect<adf::window<B*OUT_W*OUT_W*MCHUNK*4>> (convs[i].out[0], concat.in[i]);
        } else {
          adf::connect<adf::window<4>> (pin[i], concat.in[i]);
        }
      }
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (concat.out[0], pout[0]);
    }
};


#endif // __CONV_GRAPH_H__