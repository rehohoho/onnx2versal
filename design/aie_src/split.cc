#include "split.h"
#include "kernel_utils.h"


#define SPLIT_PROFILE_FOOTER(filter_name, filter_i) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d>::filter%d", \
    filter_name, typeid(TT).name(), H, INP_W, OUT_W, OVERLAP, filter_i);


#define WRITE_OUT(out, count) \
  for (int w = 0; w < count; w++) \
    window_writeincr(out, readincr(in));

#define WRITE_TWO_OUTS(prevout, nextout, count) \
  for (int w = 0; w < count; w++) { \
    TT a = readincr(in); \
    window_writeincr(prevout, a); \
    window_writeincr(nextout, a); \
  }

#define READ_IN(in, count) \
  for (int w = 0; w < count; w++) \
    readincr(in);

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter8(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1,
  output_window<TT>* out2,
  output_window<TT>* out3,
  output_window<TT>* out4,
  output_window<TT>* out5,
  output_window<TT>* out6,
  output_window<TT>* out7
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, STRIDE);
      WRITE_TWO_OUTS(out1, out2, OVERLAP);
      WRITE_OUT(out2, STRIDE);
      WRITE_TWO_OUTS(out2, out3, OVERLAP);
      WRITE_OUT(out3, STRIDE);
      WRITE_TWO_OUTS(out3, out4, OVERLAP);
      WRITE_OUT(out4, STRIDE);
      WRITE_TWO_OUTS(out4, out5, OVERLAP);
      WRITE_OUT(out5, STRIDE);
      WRITE_TWO_OUTS(out5, out6, OVERLAP);
      WRITE_OUT(out6, STRIDE);
      WRITE_TWO_OUTS(out6, out7, OVERLAP);
      WRITE_OUT(out7, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out2, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out3, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out4, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out5, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out6, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out7, OUT_W);
      READ_IN(in, INP_W - OUT_W*8+ OVERLAP*7);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 8);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter7(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1,
  output_window<TT>* out2,
  output_window<TT>* out3,
  output_window<TT>* out4,
  output_window<TT>* out5,
  output_window<TT>* out6
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, STRIDE);
      WRITE_TWO_OUTS(out1, out2, OVERLAP);
      WRITE_OUT(out2, STRIDE);
      WRITE_TWO_OUTS(out2, out3, OVERLAP);
      WRITE_OUT(out3, STRIDE);
      WRITE_TWO_OUTS(out3, out4, OVERLAP);
      WRITE_OUT(out4, STRIDE);
      WRITE_TWO_OUTS(out4, out5, OVERLAP);
      WRITE_OUT(out5, STRIDE);
      WRITE_TWO_OUTS(out5, out6, OVERLAP);
      WRITE_OUT(out6, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out2, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out3, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out4, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out5, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out6, OUT_W);
      READ_IN(in, INP_W - OUT_W*7+ OVERLAP*6);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 7);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter6(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1,
  output_window<TT>* out2,
  output_window<TT>* out3,
  output_window<TT>* out4,
  output_window<TT>* out5
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, STRIDE);
      WRITE_TWO_OUTS(out1, out2, OVERLAP);
      WRITE_OUT(out2, STRIDE);
      WRITE_TWO_OUTS(out2, out3, OVERLAP);
      WRITE_OUT(out3, STRIDE);
      WRITE_TWO_OUTS(out3, out4, OVERLAP);
      WRITE_OUT(out4, STRIDE);
      WRITE_TWO_OUTS(out4, out5, OVERLAP);
      WRITE_OUT(out5, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out2, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out3, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out4, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out5, OUT_W);
      READ_IN(in, INP_W - OUT_W*6+ OVERLAP*5);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 6);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter5(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1,
  output_window<TT>* out2,
  output_window<TT>* out3,
  output_window<TT>* out4
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, STRIDE);
      WRITE_TWO_OUTS(out1, out2, OVERLAP);
      WRITE_OUT(out2, STRIDE);
      WRITE_TWO_OUTS(out2, out3, OVERLAP);
      WRITE_OUT(out3, STRIDE);
      WRITE_TWO_OUTS(out3, out4, OVERLAP);
      WRITE_OUT(out4, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out2, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out3, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out4, OUT_W);
      READ_IN(in, INP_W - OUT_W*5+ OVERLAP*4);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 5);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter4(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1,
  output_window<TT>* out2,
  output_window<TT>* out3
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, STRIDE);
      WRITE_TWO_OUTS(out1, out2, OVERLAP);
      WRITE_OUT(out2, STRIDE);
      WRITE_TWO_OUTS(out2, out3, OVERLAP);
      WRITE_OUT(out3, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out2, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out3, OUT_W);
      READ_IN(in, INP_W - OUT_W*4 + OVERLAP*3);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 4);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter3(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1,
  output_window<TT>* out2
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, STRIDE);
      WRITE_TWO_OUTS(out1, out2, OVERLAP);
      WRITE_OUT(out2, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out2, OUT_W);
      READ_IN(in, INP_W - OUT_W*3 + OVERLAP*2);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 3);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter2(
	input_stream<TT>* in,
  output_window<TT>* out0,
  output_window<TT>* out1
) {
  PROFILE_HEADER2;

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, FIRST_STRIDE);
      WRITE_TWO_OUTS(out0, out1, OVERLAP);
      WRITE_OUT(out1, FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(out0, OUT_W);
      READ_IN(in, -OVERLAP);
      WRITE_OUT(out1, OUT_W);
      READ_IN(in, INP_W - OUT_W*2 + OVERLAP);
    }
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 2);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitScalar<TT, H, INP_W, OUT_W, OVERLAP>::filter1(
	input_stream<TT>* in,
  output_window<TT>* out0
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    WRITE_OUT(out0, OUT_W);
    READ_IN(in, INP_W - OUT_W);
  }

  SPLIT_PROFILE_FOOTER("SplitScalar", 1);
}
