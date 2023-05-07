#ifndef __CONCAT_GRAPH_H__
#define __CONCAT_GRAPH_H__

#include <adf.h>
#include "concat.h"


template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
class ConcatScalarGraph : public adf::graph {

  public:
    adf::kernel k[1];
    adf::port<input> pin[LCNT];
    adf::port<output> pout[1];

    ConcatScalarGraph() { 
      if (LCNT == 8) {
        k[0] = adf::kernel::create_object<Concat8Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 7) {
        k[0] = adf::kernel::create_object<Concat7Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 6) {
        k[0] = adf::kernel::create_object<Concat6Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 5) {
        k[0] = adf::kernel::create_object<Concat5Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 4) {
        k[0] = adf::kernel::create_object<Concat4Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 3) {
        k[0] = adf::kernel::create_object<Concat3Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 2) {
        k[0] = adf::kernel::create_object<Concat2Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      } else if (LCNT == 1) {
        k[0] = adf::kernel::create_object<Concat1Scalar<WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>>();
      }
      adf::source(k[0]) = "concat.cc";
      adf::runtime<ratio>(k[0]) = 0.6;

      for (int i = 0; i < LCNT; i++)
        adf::connect<adf::window<WINDOW_SIZE*4>> (pin[i], k[0].in[i]);
      
      // BLOCK_SIZE <= WINDOW_SIZE
      adf::connect<adf::window<WINDOW_SIZE/CHUNK_SIZE*BLOCK_SIZE*4>> (k[0].out[0], pout[0]);
    }

};


#endif // __CONCAT_GRAPH_H__