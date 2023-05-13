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


/*
WINDOW_SIZE:  size of window inputs
CHUNK_SIZE:   chunk size to concat
BLOCK_SIZE:   size of concat chunk (in case WINDOW_SIZE % CHUNK_SIZE != 0)

Standard concat on last dimension: CHUNK_SIZE = WINDOW_SIZE, BLOCK_SIZE = OUTPUT_SIZE
Concat on non-last dimension:      BLOCK_SIZE = (NLANES-1) * CHUNK_SIZE + REMAINDER

Output size = WINDOW_SIZE / CHUNK_SIZE * BLOCK_SIZE
Note BLOCK_SIZE <= WINDOW_SIZE

lenet fc1: 294 cycles
lenet fc2: 349 cycles
lenet fc3: 91 cycles
*/
template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter8(
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
    "Running Concat8Scalar<%d, %d, %d>::filter8\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
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

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter7(
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
    "Running Concat8Scalar<%d, %d, %d>::filter7\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
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

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter6(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
	input_window<float>* in5,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running Concat8Scalar<%d, %d, %d>::filter6\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
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

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter5(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
	input_window<float>* in4,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d, %d, %d>::filter5\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter4(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
	input_window<float>* in3,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d, %d, %d>::filter4\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter3(
	input_window<float>* in0,
	input_window<float>* in1,
	input_window<float>* in2,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d, %d, %d>::filter3\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter2(
	input_window<float>* in0,
	input_window<float>* in1,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d, %d, %d>::filter2\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
    int blockIdx = 0;
    CAT(in0);
    CAT(in1);
  }

  PROFILE_FOOTER;
}

template <int LCNT, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter1(
	input_window<float>* in0,
  output_window<float>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%d, %d, %d>::filter1\n", WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE));

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
    int blockIdx = 0;
    CAT(in0);
  }

  PROFILE_FOOTER;
}
