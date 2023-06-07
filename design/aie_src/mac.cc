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


template <typename TT, int B, int W, int IS_RELU>
void MacFloat<TT, B, W, IS_RELU>::filter(
	input_window<float>* in,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running MacFloat<%s,%d,%d,%d>\n", typeid(float).name(), B, W, IS_RELU));
  
  v8float data = undef_v8float();
  v8float *w_ptr = (v8float *) weights;
  v8float *b_ptr = (v8float *) bias;

  v8float zeros = null_v8float();

  v8float res = undef_v8float();
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w+=8) {
      data = window_readincr_v8(in);

      res = fpmac(*b_ptr, data, *w_ptr); w_ptr ++; b_ptr ++;
      if (IS_RELU)
        res = fpmax(res, zeros, 0, 0x76543210);
      
      window_writeincr(out, res);
    }
    w_ptr -= W/8;
    b_ptr -= W/8;
  }

  PROFILE_FOOTER;
}