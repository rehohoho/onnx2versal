#include "mac.h"
#include "kernel_utils.h"


template <typename TT, int B, int W, int IS_RELU>
void MacScalar<TT, B, W, IS_RELU>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running MacScalar<%s,%d,%d,%d>\n", typeid(TT).name(), B, W, IS_RELU));
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w++) {
      TT a = window_readincr(in);
      a = a * weights[w] + bias[w];
      if (IS_RELU)
        a = (a > 0) ? a : 0;
      window_writeincr(out, a);
    }
  }

  PROFILE_FOOTER;
}
