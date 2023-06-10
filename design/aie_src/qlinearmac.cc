#include "qlinearmac.h"
#include "kernel_utils.h"



/*
(qz-qz_zero)
= (qx-qx_zero)*qx_scale * (qw-qw_zero)*qw_scale / qz_scale 
= (qx-qx_zero) * qw *qx_scale*qw_scale/qz_scale 

(qy-qy_zero) * qy_scale
= (qz-qz_zero)*qz_scale + (qb-qb_zero)*qb_scale

*/
template <int B, int W, int IS_RELU>
void QlinearMacScalar<B, W, IS_RELU>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QlinearMacScalar<%d,%d,%d>\n", B, W, IS_RELU));
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w++) {
      int x = window_readincr(in);
      x = (x - x_zero) * (weights[w] - w_zero);
      x = round(x * x_scale*w_scale/z_scale) + z_zero;
      x = std::min(std::max(x, -128), 127);

      float z = x;
      z = (z - z_zero) * z_scale + (bias[w] - b_zero) * b_scale;
      
      int y = round(z / y_scale) + y_zero;
      y = std::min(std::max(y, -128), 127);
      if (IS_RELU)
        y = (y > 0) ? y : 0;
      window_writeincr(out, y);
    }
  }

  PROFILE_FOOTER;
}
