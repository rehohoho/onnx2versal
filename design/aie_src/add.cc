#include "add.h"
#include "kernel_utils.h"


template <typename TT, int B, int W, int IS_RELU>
void AddScalar<TT, B, W, IS_RELU>::filter(
	input_window<TT>* inA,
  input_window<TT>* inB,
  output_window<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running AddScalar<%s,%d,%d,%d>\n", typeid(TT).name(), B, W, IS_RELU));

  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w++) {
      TT c = window_readincr(inA) + window_readincr(inB);
      if (IS_RELU)
        c = (c >= 0) ? c : 0;
      window_writeincr(out, c);
    }
  }

  PROFILE_FOOTER;
}