#ifndef __QGEMM_GRAPH_H_
#define __QGEMM_GRAPH_H_

#include <adf.h>
#include "qgemm.h"
#include "graph_concat.h"
#include "graph_utils.h"


/**
 * @defgroup Qgemm
 * 
 * @brief The qgemm operator consumes a quantized input tensor, its scale and zero point, 
 * a quantized weight, its scale and zero point, and output's scale and zero point, 
 * and computes the quantized output. xA^T + b as per torch.nn.Linear. 
 * Applies general matrix multiply: output(MxN) = input(MxK) * weights(KxN) + bias(N)
 * 
 * @tparam QGEMM    QGemm Kernel
 * @tparam TT       input/output dtype, int8_t or uint8_t
 * @tparam TTPARAM  weight dtype, int8_t or uint8_t
 * @tparam M        number of rows of input matrix
 * @tparam K        number of cols / number of rows of weight matrix
 * @tparam N        number of cols of weight matrix / size of bias vector
 * 
 * @{
 */

/**
 * @brief Single instance graph that stores weights and biases
 * 
 * @connections
 * @connect{pin[0], M*K}
 * @connect{pout[0], stream M*N}
 * @endconnections
 */
template <template<typename, typename, int, int, int> class QGEMM, 
  typename TT, typename TTPARAM, int M, int K, int N>
class QgemmGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QgemmGraph(
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) { 
      k[0] = adf::kernel::create_object<QGEMM<TT, TTPARAM, M, K, N>>(
        weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qgemm.cc";
      adf::headers(k[0]) = {"qgemm.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::heap_size(k[0]) = K + 1024;

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      
      adf::samples_per_iteration(k[0].in[0]) = M*K;
      adf::samples_per_iteration(k[0].out[0]) = M*N;
    }

};



/**
 * @brief Single instance graph that stores weights and biases
 * 
 * @connections
 * @connect{pin[0], M*K}
 * @connect{pin[1], A*K*N}, where A depends on kernel 
 * @connect{pout[0], stream M*N}
 * @endconnections
 */
template <template<typename, typename, int, int, int> class QGEMM, 
  typename TT, typename TTPARAM, int M, int K, int N>
class QgemmStreamGraph : public adf::graph {

  private:
    adf::kernel k[1];

  public:
    adf::port<input> pin[2];
    adf::port<output> pout[1];

    QgemmStreamGraph(
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) { 
      k[0] = adf::kernel::create_object<QGEMM<TT, TTPARAM, M, K, N>>(
        bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
      adf::source(k[0]) = "qgemm.cc";
      adf::headers(k[0]) = {"qgemm.h"};
      adf::runtime<ratio>(k[0]) = 0.6;
      adf::heap_size(k[0]) = 24576; // assume KxN > MxN

      adf::connect<adf::stream> (pin[0], k[0].in[0]);
      adf::connect<adf::stream> (pin[1], k[0].in[1]);
      adf::connect<adf::stream> (k[0].out[0], pout[0]);
      adf::samples_per_iteration(k[0].in[0]) = M*K;
      adf::samples_per_iteration(k[0].out[0]) = M*N;

      adf::location_constraint tilePos = adf::location<adf::kernel>(k[0]);
      adf::location<adf::parameter>(k[0].param[0]) = tilePos;
      adf::location<adf::parameter>(k[0].param[0]) = adf::offset(0);
    }

};


/**
 * @brief Multiinstance graph for MxK times KxN that stores weights and biases
 * Requires KxN_RND weight, NCHUNK%8=0, N%4=0
 * Chunks KxN weights by N dimension into NCHUNK chunks.
 * Each instance has max size = 16384 and 4096 bytes respectively.
 * Places maximum of 3x3 tiles, 8 conv tiles surrounding concat tile (max AIE DMA input=8)
 */
template <
  template<typename, typename, int, int, int> class QGEMM, 
  template<typename, int, int, int, int> class CONCAT, 
  int NCHUNK, 
  typename TT, typename TTPARAM, int M, int K, int N>
class QgemmChunkNGraph : public adf::graph {

  private:

  public:
    static const int CHUNK_COUNT = (N + NCHUNK - 1) / NCHUNK; // ceiling
    adf::kernel k[CHUNK_COUNT];
    ConcatStreamGraph<CONCAT, TT, CHUNK_COUNT, M, NCHUNK, N> concat_g;
    
    adf::port<input> pin[1];
    adf::port<output> pout[1];

    QgemmChunkNGraph(
      std::vector<TTPARAM> weights,  // KxN
      std::vector<int32_t> bias,    // N
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero
    ) { 
      static_assert(CHUNK_COUNT <= 8);
      static_assert(M*K <= TILE_BYTES);
      static_assert(K*NCHUNK <= MAX_PARAM_BYTES);
      static_assert(M*NCHUNK <= TILE_BYTES);

      std::vector<int32_t> bChunk;

      for (int i = 0; i < CHUNK_COUNT; i++) {
        
        // build wchunk
        std::vector<TTPARAM> wChunk;
        wChunk.reserve(NCHUNK*K);
        for (int j = 0; j < K*N; j+=N) {
          wChunk.insert(wChunk.end(), weights.begin()+j+i*NCHUNK, weights.begin()+j+i*NCHUNK+NCHUNK);
        }
        
        // build bChunk
        bChunk = std::vector<int32_t>(bias.begin()+i*NCHUNK, bias.begin()+i*NCHUNK+NCHUNK);
        bChunk.resize(NCHUNK, 0);
        
        k[i] = adf::kernel::create_object<QGEMM<TT, TTPARAM, M, K, NCHUNK>>(
          wChunk, bChunk, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero);
        adf::source(k[i]) = "qgemm.cc";
        adf::headers(k[i]) = {"qgemm.h"};
        adf::runtime<ratio>(k[i]) = 0.6;
        adf::heap_size(k[i]) = K + 1024;

        if ((i&0x1) == 1) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
        }
        if (i == 2 || i == 6) {
          adf::location<adf::kernel>(k[i]) = adf::location<adf::kernel>(k[i-1]) + adf::relative_offset({.col_offset=0, .row_offset=2});
        }
        
        adf::location_constraint tilePos = adf::location<adf::kernel>(k[i]);
        adf::location<adf::parameter>(k[i].param[0]) = tilePos;
        adf::location<adf::parameter>(k[i].param[0]) = adf::offset(0);
        adf::location<adf::parameter>(k[i].param[1]) = tilePos;
        adf::location<adf::parameter>(k[i].param[1]) = adf::offset((K*NCHUNK+31)/32*32);
        // arbitrary input/output buffer location due to interconnect design
      }

      for (int i = 0; i < CHUNK_COUNT; i++) {
        adf::connect<adf::stream> (pin[0], k[i].in[0]);
        adf::connect<adf::stream> (k[i].out[0], concat_g.pin[i]);
        adf::samples_per_iteration(k[i].out[0]) = M*NCHUNK;
      }
      adf::connect<adf::stream> (concat_g.pout[0], pout[0]);

      for (int i = 0; i < concat_g.k1.size(); i++) {
        adf::location<adf::kernel>(concat_g.k1[i]) = 
          adf::location<adf::kernel>(k[i*2+1]) + adf::relative_offset({.col_offset=0, .row_offset=1});
      }
    }

};
/** @} */


#endif // __QGEMM_GRAPH_H_