#include "transpose.h"
#include "kernel_utils.h"


#define TRANSPOSE_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, H, W, C);

template <typename TT, int B, int H, int W, int C>
void TransposeScalarBHWC2BCHW<TT, B, H, W, C>::filter(
	input_window<TT>* in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) { 

      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          TT a = window_read(in);
          writeincr(out, a);
          window_incr(in, C);          
        } // W
        chess_separator_scheduler();
      } // H

      window_incr(in, -H*W*C + 1);
    }
  }

  TRANSPOSE_PROFILE_FOOTER("TransposeScalarBHWC2BCHW");
}


template <typename TT, int B, int H, int W, int C>
void TransposeScalarPktStreamBHWC2BCHW<TT, B, H, W, C>::filter(
	input_pktstream* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;

  for (int b = 0; b < B; b++) {
    get_ss(0);
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        
        for (int c = 0; c < C; c++) { 
          TT a = getf_ss(0);
          window_write(out, a);
          window_incr(out, H*W);  // next channel, same pos
        }
        window_incr(out, -C*H*W + 1); // reset channel, move right

      }
      // chess_separator_scheduler();
    }
  }

  TRANSPOSE_PROFILE_FOOTER("TransposeScalarBHWC2BCHW");
}
