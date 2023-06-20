#include "add.h"
#include "kernel_utils.h"


#define ADD_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d>", \
    filter_name, typeid(TT).name(), W, IS_RELU);

template <typename TT, int W, int IS_RELU>
void AddScalar<TT, W, IS_RELU>::filter(
	input_stream<TT>* restrict inA,
  input_stream<TT>* restrict inB,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  for (int w = 0; w < W; w++) {
    TT c = readincr(inA) + readincr(inB);
    if (IS_RELU)
      c = (c >= 0) ? c : 0;
    writeincr(out, c);
  }

  ADD_PROFILE_FOOTER("AddScalar");
}


template <typename TT, int W, int IS_RELU>
void AddFloat<TT, W, IS_RELU>::filter(
	input_stream<TT>* restrict inA,
  input_stream<TT>* restrict inB,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  v8float zeros = null_v8float();
  v8float av = undef_v8float();
  v8float bv = undef_v8float();

  for (int w = 0; w < W; w+=4) chess_prepare_for_pipelining chess_loop_range(W/4,) {
    av = upd_v(av, 0, getf_wss(0)); // better pipelining than loading full v8float
    bv = upd_v(bv, 0, getf_wss(1));

    av = fpadd(av, bv);
    
    if (IS_RELU)
      av = fpmax(av, zeros);
    
    put_wms(0, ext_v(av, 0));
  }

  ADD_PROFILE_FOOTER("AddFloat");
}