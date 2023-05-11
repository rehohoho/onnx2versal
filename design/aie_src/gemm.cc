#include "gemm.h"
#include "kernel_utils.h"


/*
xA^T + b as per torch,nn.Linear
8581 cycles for lenet fc1 (broke in 8 sections)
4488 cycles for lenet fc2 (broke in 4 sections)
1956 cycles for lenet fc3
MxK * NxK
weights[N*K] (120x256)
bias[N]      (120)
*/
template <int M, int K, int NCHUNK>
void GemmReluScalarMKNK<M, K, NCHUNK>::filter(
	input_window<float>* in,      // MxK  (1x256)
  output_window<float>* out     // MxN  (1x120)
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKNK<%d, %d, %d>\n", M, K, NCHUNK));

  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < NCHUNK; j++) {
      float res = bias[j];
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = weights[weightIdx];
        weightIdx++;
        res += a * b;
      }    
      
      if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next NCHUNK
  }

  PROFILE_FOOTER;
}


/*
1899 cycles (1x84 * 84x10)
*/
template <int M, int K, int NCHUNK>
void GemmReluScalarMKKN<M, K, NCHUNK>::filter(
	input_window<float>* in,      // MxK  (1x256)   inputs
                                // KxN  (256x120) weights
  output_window<float>* out     // MxN  (1x120)   outputs
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKKN<%d, %d, %d>\n", M, K, NCHUNK));

  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < NCHUNK; j++) {
      float res = bias[j];
      weightIdx = j;
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = weights[weightIdx];
        weightIdx += NCHUNK_RND;
        res += a * b;
      }    
      if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next NCHUNK
  }

  PROFILE_FOOTER;
}


/*
Using:
v8float fpmac (v8float        acc,
		           v16float       xbuf,
               int  	        xstart,
               unsigned int  	xoffs,
               v8float  	    zbuf, 
               int  	        zstart, !! compile time constant if zbuf !!
               unsigned int  	zoffs)

for (i = 0; i < 8; i++)
  ret[i] = acc[i] + xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]]

a0a1a7 b00 b01 b07 b08 b09 b0f
       b10 b11 b17 b17 b19 b1f
       b70 b71 b77 b77 b79 b7f

a0 * b00 b01 ... b07
393 cycles (1x84 * 84x10)
*/
template <int M, int K, int NCHUNK> // K%4=0, N%4=0
void GemmReluMKKN<M, K, NCHUNK>::filter(
	input_window<float>* in,      // MxK  (1x256)   inputs
                                // KxN  (256x120) weights
  output_window<float>* out     // MxN  (1x120)   outputs
) {
  PROFILE_HEADER(printf(
    "Running GemmReluMKKN<%d, %d, %d>\n", M, K, NCHUNK));

  float *a_ptr = (float *) in->ptr;
  float *w_ptr = (float *) weights;
  float *b_ptr = (float *) bias;
  v8float zeros = null_v8float();
  v8float matA = undef_v8float();
  v8float matB = undef_v8float();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < NCHUNK; j+=8) { // fpmac accsize 8

      v8float acc1 = *(v8float *) (b_ptr + j);

      for (int k = 0; k < K; k+=4) {
        matA = *(v8float *) a_ptr;
        matB = *(v8float *) w_ptr; w_ptr += NCHUNK_RND;
        acc1 = fpmac(acc1, matA, 0, 0x00000000, matB, 0, 0x76543210);
        matB = *(v8float *) w_ptr; w_ptr += NCHUNK_RND;
        acc1 = fpmac(acc1, matA, 1, 0x00000000, matB, 0, 0x76543210);
        matB = *(v8float *) w_ptr; w_ptr += NCHUNK_RND;
        acc1 = fpmac(acc1, matA, 2, 0x00000000, matB, 0, 0x76543210);
        matB = *(v8float *) w_ptr; w_ptr += NCHUNK_RND;
        acc1 = fpmac(acc1, matA, 3, 0x00000000, matB, 0, 0x76543210);
        a_ptr += 4;
      }
      acc1 = fpmax(acc1, zeros, 0, 0x76543210);

      if (NCHUNK - j < 8) {
        float *acc_ptr = (float *) &acc1;
        for (int i = 0; i < NCHUNK - j; i++) {
          window_writeincr(out, acc_ptr[i]);
        }
      } else {
        window_writeincr(out, acc1);
      }

      w_ptr += -K*NCHUNK_RND + 8;
      a_ptr -= K;
    }
    w_ptr -= NCHUNK_RND;
    a_ptr += K;
    
  }
  

  PROFILE_FOOTER;
}
