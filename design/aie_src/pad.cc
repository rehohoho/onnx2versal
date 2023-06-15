#include "pad.h"
#include "kernel_utils.h"


#define WRITE_ZERO(out, len) \
  for (int i = 0; i < len; i++) writeincr(out, 0);

template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DScalar<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER(printf(
    "Running Pad2DScalar::filter<%s,%d,%d,%d,%d,%d,%d,%d>\n", typeid(TT).name(), B, INP_H, INP_W, H0, H1, W0, W1));
  
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

  PROFILE_FOOTER;
}
