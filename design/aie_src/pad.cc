#include "pad.h"
#include "kernel_utils.h"


template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DScalar<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_window<TT>* in,  // NxINP_W
  output_window<TT>* out // NxOUT_W
) {
  PROFILE_HEADER(printf(
    "Running Pad2DScalar::filter<%s,%d,%d,%d,%d,%d,%d,%d>\n", typeid(TT).name(), B, INP_H, INP_W, H0, H1, W0, W1));
  
  for (int b = 0; b < B; b++) {
    window_incr(out, H0*OUT_W);
    
    for (int h = 0; h < INP_H; h++) {
      window_incr(out, W0);

      for (int w = 0; w < INP_W; w++) {
        window_writeincr(out, window_readincr(in));
      }

      window_incr(out, W1);
    }

    window_incr(out, H1*OUT_W);
  }

  PROFILE_FOOTER;
}
