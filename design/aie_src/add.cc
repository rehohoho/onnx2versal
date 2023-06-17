#include "add.h"
#include "kernel_utils.h"


template <typename TT, int W, int IS_RELU>
void AddScalar<TT, W, IS_RELU>::filter(
	input_stream<TT>* restrict inA,
  input_stream<TT>* restrict inB,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER(printf(
    "Running AddScalar<%s,%d,%d>\n", typeid(TT).name(), W, IS_RELU));

  for (int w = 0; w < W; w++) {
    TT c = readincr(inA) + readincr(inB);
    if (IS_RELU)
      c = (c >= 0) ? c : 0;
    writeincr(out, c);
  }

  PROFILE_FOOTER;
}


template <typename TT, int W, int IS_RELU>
void AddFloat<TT, W, IS_RELU>::filter(
	input_stream<TT>* restrict inA,
  input_stream<TT>* restrict inB,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER(printf(
    "Running AddFloat<%s,%d,%d>\n", typeid(TT).name(), W, IS_RELU));

  v8float zeros = null_v8float();
  v8float av = undef_v8float();
  v8float bv = undef_v8float();

  for (int w = 0; w <= W-8; w+=8) chess_prepare_for_pipelining chess_loop_range(W/8,) { // W%8
    av = upd_v(av, 0, getf_wss(0));
    bv = upd_v(bv, 0, getf_wss(1));
    av = upd_v(av, 1, getf_wss(0));
    bv = upd_v(bv, 1, getf_wss(1));

    av = fpadd(av, bv);
    
    if (IS_RELU)
      av = fpmax(av, zeros);
    
    put_wms(0, ext_v(av, 0));
    put_wms(0, ext_v(av, 1));
  }

  if (W % 8 == 4) { // W%4
    av = upd_v(av, 0, getf_wss(0));
    bv = upd_v(bv, 0, getf_wss(1));
    av = fpadd(av, bv);
    
    if (IS_RELU)
      av = fpmax(av, zeros);
    
    put_wms(0, ext_v(av, 0));
  }

  PROFILE_FOOTER;
}