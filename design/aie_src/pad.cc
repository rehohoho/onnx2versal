#include "pad.h"
#include "kernel_utils.h"


template <typename TT, int N, int INP_W, int OUT_W>
void PadScalar<TT, N, INP_W, OUT_W>::filter(
	input_window<TT>* in,  // NxINP_W
  output_window<TT>* out // NxOUT_W
) {
  PROFILE_HEADER(printf(
    "Running PadScalar::filter<%d, %d>\n", INP_W, OUT_W));
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < INP_W; j++) {
      TT a = window_readincr(in);
      window_writeincr(out, a);
    }
    window_incr(out, OUT_W - INP_W);
  }

  PROFILE_FOOTER;
}
