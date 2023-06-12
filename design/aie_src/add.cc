#include "add.h"
#include "kernel_utils.h"


template <typename TT, int W, int IS_RELU>
void AddScalar<TT, W, IS_RELU>::filter(
	input_window<TT>* inA,
  input_window<TT>* inB,
  output_window<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running AddScalar<%s,%d,%d>\n", typeid(TT).name(), W, IS_RELU));

  for (int w = 0; w < W; w++) {
    TT c = window_readincr(inA) + window_readincr(inB);
    if (IS_RELU)
      c = (c >= 0) ? c : 0;
    window_writeincr(out, c);
  }

  PROFILE_FOOTER;
}


template <typename TT, int W, int IS_RELU>
void AddFloat<TT, W, IS_RELU>::filter(
	input_window<TT>* inA,
  input_window<TT>* inB,
  output_window<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running AddFloat<%s,%d,%d>\n", typeid(TT).name(), W, IS_RELU));

  v8float zeros = null_v8float();
  v8float av = undef_v8float();
  v8float bv = undef_v8float();

  for (int w = 0; w <= W-8; w+=8) { // W%8
    av = window_readincr_v8(inA); // limited by read bandwidth
    bv = window_readincr_v8(inB);
    av = fpadd(av, bv);
    
    if (IS_RELU)
      av = fpmax(av, zeros);
    window_writeincr(out, av);
  }

  if (W % 8 == 4) { // W%4
    av = upd_v(av, 0, window_readincr_v4(inA));
    bv = upd_v(bv, 0, window_readincr_v4(inB));
    av = fpadd(av, bv);
    
    if (IS_RELU)
      av = fpmax(av, zeros);
    window_writeincr(out, ext_v(av, 0));
  }

  PROFILE_FOOTER;
}