#include "split.h"
#include "kernel_utils.h"


#define SPLIT_PROFILE_FOOTER(filter_name, filter_i) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d>::filter%d", \
    filter_name, typeid(TT).name(), H, INP_W, OUT_W, OVERLAP, filter_i);


// SplitScalar
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

#undef WRITE_OUT
#undef WRITE_TWO_OUTS
#undef READ_IN


// SplitInt8
#define WRITE_OUT(out, count) \
  for (int w = 0; w < count; w+=16) { \
    window_writeincr(out, readincr_v16(in)); \
  }

#define WRITE_TWO_OUTS(prevout, nextout, count) \
  for (int w = 0; w < count; w+=16) { \
    auto a = readincr_v16(in); \
    window_writeincr(prevout, a); \
    window_writeincr(nextout, a); \
  }

#define READ_IN(in, count) \
  for (int w = 0; w < count; w+=16) \
    readincr_v16(in);

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter8(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 8);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter7(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 7);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter6(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 6);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter5(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 5);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter4(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 4);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter3(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 3);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter2(
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

  SPLIT_PROFILE_FOOTER("SplitInt8", 2);
}

template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitInt8<TT, H, INP_W, OUT_W, OVERLAP>::filter1(
	input_stream<TT>* in,
  output_window<TT>* out0
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    WRITE_OUT(out0, OUT_W);
    READ_IN(in, INP_W - OUT_W);
  }

  SPLIT_PROFILE_FOOTER("SplitInt8", 1);
}

#undef WRITE_OUT
#undef WRITE_TWO_OUTS
#undef READ_IN


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitTwo32bitStreams<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_stream<TT>* restrict out0,
  output_stream<TT>* restrict out1
) {
  PROFILE_HEADER2;

  TT a;

#define WRITE_OUT(out_idx, count) \
  for (int w = 0; w < count; w++) { \
    a = get_ss(0); \
    put_ms(out_idx, a); \
  }

#define WRITE_TWO_OUTS(count) \
  for (int w = 0; w < count; w++) { \
    a = get_ss(0); \
    put_ms(0, a); \
    put_ms(1, a); \
  }

#define READ_IN(count) \
  for (int w = 0; w < count; w++) \
    get_ss(0);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      WRITE_OUT(0, OVERLAP);
      WRITE_OUT(0, STRIDE);
      
      for (int i = 1; i < LCNT; i++) {
        WRITE_TWO_OUTS(OVERLAP);
        WRITE_OUT((i & 0x1), STRIDE);
      }

      WRITE_OUT(((LCNT-1) & 0x1), OVERLAP);
    }
  } else {
    for (int h = 0; h < H; h++) {
      for (int i = 0; i < LCNT; i++) {
        WRITE_OUT((i & 0x1), OUT_W);
        READ_IN(-OVERLAP);
      }
      READ_IN(INP_W - OUT_W*2 + 2*OVERLAP);
    }
  }

#undef WRITE_OUT
#undef WRITE_TWO_OUTS
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitTwo32bitStreams", 2);
}


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitFilterFloatStream<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_stream<TT>* out0
) {
  PROFILE_HEADER2;

  TT a;
  int pre_read_lanes = lane_idx;
  int post_read_lanes = LCNT - lane_idx - 1;

// 32-bit read / cycle or 128-bit read / 4 cycle
#define WRITE_OUT(count) \
  for (int w = 0; w < count; w++) \
    put_ms(0, get_ss(0));

#define READ_IN(count) \
  for (int w = 0; w < count; w++) \
    get_ss(0);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * FIRST_STRIDE);
      WRITE_OUT(OUT_W);
      READ_IN(post_read_lanes * FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * (OUT_W - OVERLAP));
      WRITE_OUT(OUT_W);
      READ_IN(post_read_lanes * (OUT_W - OVERLAP));
    }
  }

#undef WRITE_OUT
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitFilterFloatStream", lane_idx);
}


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitFilterFloatStreamTwice<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_stream<TT>* out0,
  output_stream<TT>* out1
) {
  PROFILE_HEADER2;

  v4float vec;
  int pre_read_lanes = lane_idx;
  int post_read_lanes = LCNT - lane_idx - 2;

// 32-bit read / cycle or 128-bit read / 4 cycle
#define WRITE_OUT(outidx, count) \
  for (int w = 0; w < count; w++) \
    put_ms(outidx, get_ss(0));

#define WRITE_OUT_TWICE(count) \
  for (int w = 0; w < count; w++) { \
    auto a = get_ss(0); \
    put_ms(0, a); \
    put_ms(1, a); \
  }

#define READ_IN(count) \
  for (int w = 0; w < count; w++) \
    get_ss(0);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * FIRST_STRIDE);
      WRITE_OUT(0, FIRST_STRIDE);
      WRITE_OUT_TWICE(OVERLAP);
      WRITE_OUT(1, FIRST_STRIDE);
      READ_IN(post_read_lanes * FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * (OUT_W - OVERLAP));
      WRITE_OUT(0, OUT_W);
      READ_IN(- OVERLAP);
      WRITE_OUT(1, OUT_W);
      READ_IN(post_read_lanes * (OUT_W - OVERLAP));
    }
  }

#undef WRITE_OUT
#undef WRITE_OUT_TWICE
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitFilterFloatStreamTwice", lane_idx);
}


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitFilterFloatPktStream<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_pktstream* out0,
  output_pktstream* out1
) {
  PROFILE_HEADER2;

  uint32 ID[LCNT];
  output_pktstream *out[2] = {out0, out1};

  for (int i = 0; i < (LCNT+1)/2; i++)
    ID[2*i] = getPacketid(out0, i);
  for (int i = 0; i < LCNT/2; i++)
    ID[2*i+1] = getPacketid(out1, i);

// 32-bit read / cycle or 128-bit read / 4 cycle
#define WRITE_OUT(outidx, count, tlast) \
  for (int w = 0; w < count - 1; w++) { \
    float a = getf_ss(0); \
    writeincr(out[outidx], a); \
  } \
  writeincr(out[outidx], getf_ss(0), tlast);

#define WRITE_OUT_TWICE(outidx0, outidx1, count) \
  for (int w = 0; w < count - 1; w++) { \
    auto a = getf_ss(0); \
    writeincr(out[outidx0], a); \
    writeincr(out[outidx1], a); \
  } \
  auto a = getf_ss(0); \
  writeincr(out[outidx0], a, true); \
  writeincr(out[outidx1], a);

#define READ_IN(count) \
  for (int w = 0; w < count; w++) \
    get_ss(0);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) chess_prepare_for_pipelining chess_loop_range(H,) {

      writeHeader(out[0], 0, ID[0]);
      WRITE_OUT(0, FIRST_STRIDE, false);

      for (int i = 1; i < LCNT; i++) chess_flatten_loop {
        writeHeader(out[i&0x1], 0, ID[i]);
        WRITE_OUT_TWICE(1 - (i&0x1), i&0x1, OVERLAP);
        WRITE_OUT(i&0x1, STRIDE, false);
      } // LCNT

      int i = LCNT - 1;
      WRITE_OUT(i&0x1, OVERLAP, true);
    } // H

  } else {
    for (int h = 0; h < H; h++) chess_prepare_for_pipelining chess_loop_range(H,) {

      for (int i = 0; i < LCNT-1; i++) chess_flatten_loop {
        writeHeader(out[i&0x1], 0, ID[i]);
        WRITE_OUT(i&0x1, OUT_W, true);
        READ_IN(-OVERLAP);
      } // LCNT

      int i = LCNT - 1;
      writeHeader(out[i&0x1], 0, ID[i]);
      WRITE_OUT(i&0x1, OUT_W, true);
      READ_IN(INP_W - OUT_W*LCNT + OVERLAP*(LCNT-1));
    } // H 
  }

#undef WRITE_OUT
#undef WRITE_OUT_TWICE
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitFilterFloatPktStream", LCNT);
}


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitFilterInt8Stream<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_stream<TT>* out0
) {
  PROFILE_HEADER2;

  TT a;
  int pre_read_lanes = lane_idx;
  int post_read_lanes = LCNT - lane_idx - 1;

// 32-bit read / cycle or 128-bit read / 4 cycle
#define WRITE_OUT(count) \
  for (int w = 0; w < count; w+=16) \
    writeincr_v16(out0, readincr_v16(in));

#define READ_IN(count) \
  for (int w = 0; w < count; w+=16) \
    readincr_v16(in);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * FIRST_STRIDE);
      WRITE_OUT(OUT_W);
      READ_IN(post_read_lanes * FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * (OUT_W - OVERLAP));
      WRITE_OUT(OUT_W);
      READ_IN(post_read_lanes * (OUT_W - OVERLAP));
    }
  }

#undef WRITE_OUT
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitFilterInt8Stream", lane_idx);
}


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitFilterInt8StreamTwice<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_stream<TT>* out0,
  output_stream<TT>* out1
) {
  PROFILE_HEADER2;

  output_stream<TT> *out[2] = {out0, out1};

  v4float vec;
  int pre_read_lanes = lane_idx;
  int post_read_lanes = LCNT - lane_idx - 2;

// 32-bit read / cycle or 128-bit read / 4 cycle
#define WRITE_OUT(outidx, count) \
  for (int w = 0; w < count; w+=16) \
    writeincr_v16(out[outidx], readincr_v16(in));

#define WRITE_OUT_TWICE(count) \
  for (int w = 0; w < count; w+=16) { \
    auto a = readincr_v16(in); \
    writeincr_v16(out0, a); \
    writeincr_v16(out1, a); \
  }

#define READ_IN(count) \
  for (int w = 0; w < count; w+=16) \
    readincr_v16(in);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * FIRST_STRIDE);
      WRITE_OUT(0, FIRST_STRIDE);
      WRITE_OUT_TWICE(OVERLAP);
      WRITE_OUT(1, FIRST_STRIDE);
      READ_IN(post_read_lanes * FIRST_STRIDE);
    }
  } else {
    for (int h = 0; h < H; h++) {
      READ_IN(pre_read_lanes * (OUT_W - OVERLAP));
      WRITE_OUT(0, OUT_W);
      READ_IN(- OVERLAP);
      WRITE_OUT(1, OUT_W);
      READ_IN(post_read_lanes * (OUT_W - OVERLAP));
    }
  }

#undef WRITE_OUT
#undef WRITE_OUT_TWICE
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitFilterInt8StreamTwice", lane_idx);
}


template <typename TT, int H, int INP_W, int OUT_W, int OVERLAP>
void SplitFilterInt8PktStream<TT, H, INP_W, OUT_W, OVERLAP>::filter(
	input_stream<TT>* in,
  output_pktstream* out0,
  output_pktstream* out1
) {
  PROFILE_HEADER2;

  uint32 ID[LCNT];
  output_pktstream *out[2] = {out0, out1};

  for (int i = 0; i < (LCNT+1)/2; i++)
    ID[2*i] = getPacketid(out0, i);
  for (int i = 0; i < LCNT/2; i++)
    ID[2*i+1] = getPacketid(out1, i);

// 32-bit read / cycle or 128-bit read / 4 cycle
#define WRITE_OUT(outidx, count, tlast) \
  for (int w = 0; w < count; w+=16) { \
    auto a = readincr_v16(in); \
    put_wms(outidx, a); \
  } \
  if (tlast) writeincr(out[outidx], 0, tlast);

#define WRITE_OUT_TWICE(outidx0, outidx1, count) \
  for (int w = 0; w < count; w+=16) { \
    auto a = readincr_v16(in); \
    put_wms(0, a); \
    put_wms(1, a); \
  } \
  writeincr(out[outidx0], 0, true);

#define READ_IN(count) \
  for (int w = 0; w < count; w+=16) \
    readincr_v16(in);

  if (OVERLAP > 0) {
    for (int h = 0; h < H; h++) chess_prepare_for_pipelining chess_loop_range(H,) {

      writeHeader(out[0], 0, ID[0]);
      WRITE_OUT(0, FIRST_STRIDE, 0);

      for (int i = 1; i < LCNT; i++) chess_flatten_loop {
        writeHeader(out[i&0x1], 0, ID[i]);
        WRITE_OUT_TWICE(1 - (i&0x1), i&0x1, OVERLAP);
        WRITE_OUT(i&0x1, STRIDE, 0);
      } // LCNT

      int i = LCNT - 1;
      WRITE_OUT(i&0x1, OVERLAP, 1);
    } // H

  } else {
    for (int h = 0; h < H; h++) chess_prepare_for_pipelining chess_loop_range(H,) {

      for (int i = 0; i < LCNT-1; i++) chess_flatten_loop {
        writeHeader(out[i&0x1], 0, ID[i]);
        WRITE_OUT(i&0x1, OUT_W, 1);
        READ_IN(-OVERLAP);
      } // LCNT

      int i = LCNT - 1;
      writeHeader(out[i&0x1], 0, ID[i]);
      WRITE_OUT(i&0x1, OUT_W, 1);
      READ_IN(INP_W - OUT_W*LCNT + OVERLAP*(LCNT-1));
    } // H 
  }

#undef WRITE_OUT
#undef WRITE_OUT_TWICE
#undef READ_IN

  SPLIT_PROFILE_FOOTER("SplitFilterInt8PktStream", LCNT);
}