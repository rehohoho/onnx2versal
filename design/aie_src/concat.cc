#include "concat.h"
#include "kernel_utils.h"


#define CAT(INP_WIN) \
  for (int i = 0; i < INP_W; i++) { \
    if (outi < OUT_W) { \
      float a = window_readincr(INP_WIN); \
      window_writeincr(out, a); \
    } else { \
      window_incr(INP_WIN, 1); \
    } \
    outi++; }


template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter8(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
	input_window<float>* in6,
	input_window<float>* in7,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter8\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    CAT(in6);
    CAT(in7);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter7(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
	input_window<float>* in6,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter7\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    CAT(in6);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter6(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter6\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter5(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter5\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter4(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter4\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter3(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter3\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter2(
	input_window<float>* in0,
	input_window<float>* in1,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter2\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<LCNT, H, INP_W, OUT_W>::filter1(
	input_window<float>* in0,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter1\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}


// assumes INP_W%8=0, OUT_W%4=0 (vector writes)
#define CAT_VEC(INP_WIN) \
  if (outi + INP_W <= OUT_W) { \
    for (int i = 0; i < INP_W; i+=8) \
      window_writeincr(out, window_readincr_v8(INP_WIN)); \
  } else if (outi < OUT_W) { \
    for (int i = 0; i < OUT_W - outi - 7; i+=8) \
      window_writeincr(out, window_readincr_v8(INP_WIN)); \
    for (int i = 0; i < (OUT_W - outi) % 8; i++) \
      window_writeincr(out, window_readincr(INP_WIN)); \
    window_incr(INP_WIN, outi + INP_W - OUT_W); \
  } \
  outi += INP_W;

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter8(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
	input_window<float>* in6,
	input_window<float>* in7,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter8\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    CAT_VEC(in5);
    CAT_VEC(in6);
    CAT_VEC(in7);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter7(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
	input_window<float>* in6,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter7\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    CAT_VEC(in5);
    CAT_VEC(in6);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter6(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter6\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    CAT_VEC(in5);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter5(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter5\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter4(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter4\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter3(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter3\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter2(
	input_window<float>* in0,
	input_window<float>* in1,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter2\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int H, int INP_W, int OUT_W>
void ConcatVector<LCNT, H, INP_W, OUT_W>::filter1(
	input_window<float>* in0,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter7\n", LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_VEC(in0);
    if (OUT_W > INP_W*LCNT) window_incr(out, OUT_W - LCNT*INP_W);
  }

  PROFILE_FOOTER;
}
