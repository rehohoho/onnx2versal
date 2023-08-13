#include "transpose.h"
#include "kernel_utils.h"


#define TRANSPOSE_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, H, W, C, PAD_W);


template <typename TT, int B, int H, int W, int C, int PAD_W>
void TransposeScalarBHWC2BCHW<TT, B, H, W, C, PAD_W>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;

  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) {

      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          TT a = window_read(in);
          window_writeincr(out, a);
          window_incr(in, C);
        } // W
      } // H

      window_incr(in, -H*W*C + 1);
    } // C
  }

  TRANSPOSE_PROFILE_FOOTER("TransposeScalarBHWC2BCHW");
}


template <typename TT, int B, int H, int W, int C, int PAD_W>
void TransposeScalarBCHW2BHWC<TT, B, H, W, C, PAD_W>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;

  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          TT a = window_read(in);
          window_writeincr(out, a);
          window_incr(in, H*PAD_W);
        } // C
        window_incr(in, -C*H*PAD_W + 1);
      } // W
      window_incr(in, PAD_W - W);
    } // H
  }

  TRANSPOSE_PROFILE_FOOTER("TransposeScalarBCHW2BHWC");
}


template <typename TT, int B, int H, int W, int C, int PAD_W>
void TransposeScalarBHWC2BCHWStream<TT, B, H, W, C, PAD_W>::filter(
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

  TRANSPOSE_PROFILE_FOOTER("TransposeScalarBHWC2BCHWStream");
}


template <typename TT, int B, int H, int W, int C, int PAD_W>
void TransposeScalarPktStreamBHWC2BCHW<TT, B, H, W, C, PAD_W>::filter(
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
