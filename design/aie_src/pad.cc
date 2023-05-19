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
      window_writeincr(out, window_readincr(in));
    }
    window_incr(out, OUT_W - INP_W);
  }

  PROFILE_FOOTER;
}


template <typename TT, int N, int INP_W, int OUT_W>
void PadVector<TT, N, INP_W, OUT_W>::filter(
	input_window<TT>* in,  // NxINP_W
  output_window<TT>* out // NxOUT_W
) {
  PROFILE_HEADER(printf(
    "Running PadVector::filter<%d, %d>\n", INP_W, OUT_W));
  
  const int word_size = WORD_SIZE_BITS/sizeof(TT);

  v8int16 v = null_v8int16();
  
  for (int i = 0; i < N; i++) {
    int j = 0;
    for (j; j < INP_W-7; j+=8) {
      for (int k = 0; k < 8; k++) {
        v = upd_elem(v, k, window_readincr(in));
      }
      window_writeincr(out, v);
    }
    for (j; j < INP_W; j++) {
      window_writeincr(out, window_readincr(in));
    }
    window_incr(out, OUT_W-INP_W);
  }

  PROFILE_FOOTER;
}
