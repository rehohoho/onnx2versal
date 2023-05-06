#include "transpose.h"
#include "kernel_utils.h"


template <int B, int H, int W, int C>
void TransposeScalarBHWC2BCHW<B, H, W, C>::filter(
	input_window<float>* in,      // BHWC (1x4x4x16)
  output_window<float>* out     // BCHW (1x16x4x4)
) {
  PROFILE_HEADER(printf(
    "Running TransposeScalarBHWC2BCHW::filter<%d, %d, %d, %d>\n", B, H, W, C));

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        
        for (int c = 0; c < C; c++) { 
          float a = window_readincr(in);
          window_write(out, a);
          window_incr(out, H*W);  // next channel, same pos
        }
        window_incr(out, -C*H*W + 1); // reset channel, move right

      }
    }
  }

  PROFILE_FOOTER;
}
