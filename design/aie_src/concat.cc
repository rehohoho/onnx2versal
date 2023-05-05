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
NLANES:       valid window inputs
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
template <int NLANES, int WINDOW_SIZE, int CHUNK_SIZE, int BLOCK_SIZE>
void ConcatScalar<NLANES, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>::filter(
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
  PROFILE_HEADER;
  printf("Running concat8_scalar<%d, %d, %d, %d>\n", NLANES, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE);

  for (int i = 0; i < WINDOW_SIZE; i+=CHUNK_SIZE) {
    int blockIdx = 0;
    if (NLANES >= 1) CAT(in0);
    if (NLANES >= 2) CAT(in1);
    if (NLANES >= 3) CAT(in2);
    if (NLANES >= 4) CAT(in3);
    if (NLANES >= 5) CAT(in4);
    if (NLANES >= 6) CAT(in5);
    if (NLANES >= 7) CAT(in6);
    if (NLANES >= 8) CAT(in7);
  }

  PROFILE_FOOTER;
}