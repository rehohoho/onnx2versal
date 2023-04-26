#include "transpose.h"
#include "kernel_utils.h"


template <int B, int H, int W, int C>
void bhwc2bchw_scalar(
	input_window<float>* in,      // BHWC (1x4x4x16)
  output_window<float>* out     // BCHW (1x16x4x4)
) {

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

}
