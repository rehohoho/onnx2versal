#include "pad.h"
#include "kernel_utils.h"


#define PAD_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, INP_H, INP_W, H0, H1, W0, W1);

template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DStreamScalar<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

#define WRITE_ZERO(out, len) \
  for (int i = 0; i < len; i++) writeincr(out, 0);

  for (int b = 0; b < B; b++) {
    WRITE_ZERO(out, H0*OUT_W);
    
    for (int h = 0; h < INP_H; h++) {
      WRITE_ZERO(out, W0);

      for (int w = 0; w < INP_W; w++) {
        writeincr(out, readincr(in));
      }

      WRITE_ZERO(out, W1);
    }

    WRITE_ZERO(out, H1*OUT_W);
  }

#undef WRITE_ZERO

  PAD_PROFILE_FOOTER("Pad2DStreamScalar");
}


// bandwidth limited, not much speedup
template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DStreamFloat<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  v4float a = undef_v4float();
  v4float zero = null_v4float();

#define WRITE_ZERO(out, len) \
  for (int i = 0; i <= len-4; i+=4) put_wms(0, zero); \
  for (int i = 0; i < len%4; i++) put_ms(0, 0);

  for (int b = 0; b < B; b++) {
    WRITE_ZERO(out, H0*OUT_W);
    
    for (int h = 0; h < INP_H; h++) {
      WRITE_ZERO(out, W0);

      for (int w = 0; w < INP_W; w+=4) {
        a = getf_wss(0);
        put_wms(0, a);
      }

      WRITE_ZERO(out, W1);
    }

    WRITE_ZERO(out, H1*OUT_W);
  }

#undef WRITE_ZERO

  PAD_PROFILE_FOOTER("Pad2DStreamFloat");
}


// stream int8 requires shuffle since bitwidth=32 or 128
template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DWindowScalar<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_window<TT>* restrict in,
  output_window<TT>* restrict out
) {
  PROFILE_HEADER2;

  window_incr(out, H0*OUT_W + W0);
  
  for (int h = 0; h < INP_H; h++) {

    for (int w = 0; w < INP_W; w++) {
      TT a = window_readincr(in);
      window_writeincr(out, a);
    }

    window_incr(out, W0+W1);
  }

  window_incr(out, H1*OUT_W - W0);

  PAD_PROFILE_FOOTER("Pad2DWindowScalar");
}