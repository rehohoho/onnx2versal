#include "concat.h"
#include "kernel_utils.h"


#define CAT(INP_WIN) \
  for (int i = 0; i < CHUNK_SIZE; i++) { \
    if (blockIdx < BLOCK_SIZE) { \
      float a = window_readincr(INP_WIN); \
      window_writeincr(out, a); \
    } else { \
      window_incr(INP_WIN, 1); \
    } \
    blockIdx++; }


template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter8(
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
    "Running ConcatScalar<%d,%d,%d,%d>::filter8\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    CAT(in6);
    CAT(in7);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter7(
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
    "Running ConcatScalar<%d,%d,%d,%d>::filter7\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    CAT(in6);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter6(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter6\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter5(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter5\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter4(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter4\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter3(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter3\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter2(
	input_window<float>* in0,
	input_window<float>* in1,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter2\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter1(
	input_window<float>* in0,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d,%d,%d,%d>::filter1\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT(in0);
  }

  PROFILE_FOOTER;
}


// assumes CHUNK_SIZE%8=0, BLOCK_SIZE%4=0 (vector writes)
#define CAT_VEC(INP_WIN) \
  if (blockIdx + CHUNK_SIZE <= BLOCK_SIZE) { \
    for (int i = 0; i < CHUNK_SIZE; i+=8) \
      window_writeincr(out, window_readincr_v8(INP_WIN)); \
  } else if (blockIdx < BLOCK_SIZE) { \
    for (int i = 0; i < BLOCK_SIZE - blockIdx - 7; i+=8) \
      window_writeincr(out, window_readincr_v8(INP_WIN)); \
    for (int i = 0; i < (BLOCK_SIZE - blockIdx) % 8; i++) \
      window_writeincr(out, window_readincr(INP_WIN)); \
    window_incr(INP_WIN, blockIdx + CHUNK_SIZE - BLOCK_SIZE); \
  } \
  blockIdx += CHUNK_SIZE;

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter8(
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
    "Running ConcatVector<%d,%d,%d,%d>::filter8\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    CAT_VEC(in5);
    CAT_VEC(in6);
    CAT_VEC(in7);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter7(
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
    "Running ConcatVector<%d,%d,%d,%d>::filter7\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    CAT_VEC(in5);
    CAT_VEC(in6);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter6(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter6\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
    CAT_VEC(in5);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter5(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter5\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
    CAT_VEC(in4);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter4(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter4\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
    CAT_VEC(in3);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter3(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter3\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
    CAT_VEC(in2);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter2(
	input_window<float>* in0,
	input_window<float>* in1,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter2\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
    CAT_VEC(in1);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int CHUNK_CNT, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatVector<LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE>::filter1(
	input_window<float>* in0,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatVector<%d,%d,%d,%d>::filter7\n", LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < CHUNK_CNT; i++) {
    int blockIdx = 0;
    CAT_VEC(in0);
  }

  PROFILE_FOOTER;
}
