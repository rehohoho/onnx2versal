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

1x84 * 84x10:
  single acc/fpmac: 393 cycles, no fpmac interleaving
  upd_w > load v16 from pointer, allow interleaving: 406 -> 698
  interleaving loads: 406 -> 365

Typically K > N for downsampling, M=1 if each net does an instance
If chunking by N, for N%16=0, K<=128, for N%8=0, K<=256
*/
template <int M, int K, int NCHUNK> // K%4=0, N%16=0
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
  v16float matB = null_v16float();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < NCHUNK; j+=16) { // fpmac accsize 8, do 2

      v8float acc1 = *(v8float *) (b_ptr + j);
      v8float acc2 = *(v8float *) (b_ptr + j + 8);

      for (int k = 0; k < K; k+=4) {
        matA = *(v8float *) a_ptr; a_ptr += 4;
        matB = upd_w(matB, 0, *(v8float*) w_ptr);
        acc1 = fpmac(acc1, matB, 0, 0x76543210, matA, 0, 0x00000000);
        matB = upd_w(matB, 1, *(v8float*) (w_ptr + 8)); w_ptr += NCHUNK_RND;
        // print_fvec<float>((float *) &matB, 16);
        acc2 = fpmac(acc2, matB, 8, 0x76543210, matA, 0, 0x00000000);
        matB = upd_w(matB, 0, *(v8float*) w_ptr);
        acc1 = fpmac(acc1, matB, 0, 0x76543210, matA, 1, 0x00000000);
        matB = upd_w(matB, 1, *(v8float*) (w_ptr + 8)); w_ptr += NCHUNK_RND;
        // print_fvec<float>((float *) &matB, 16);
        acc2 = fpmac(acc2, matB, 8, 0x76543210, matA, 1, 0x00000000);
        matB = upd_w(matB, 0, *(v8float*) w_ptr);
        acc1 = fpmac(acc1, matB, 0, 0x76543210, matA, 2, 0x00000000);
        matB = upd_w(matB, 1, *(v8float*) (w_ptr + 8)); w_ptr += NCHUNK_RND;
        // print_fvec<float>((float *) &matB, 16);
        acc2 = fpmac(acc2, matB, 8, 0x76543210, matA, 2, 0x00000000);
        matB = upd_w(matB, 0, *(v8float*) w_ptr);
        acc1 = fpmac(acc1, matB, 0, 0x76543210, matA, 3, 0x00000000);
        matB = upd_w(matB, 1, *(v8float*) (w_ptr + 8)); w_ptr += NCHUNK_RND;
        // print_fvec<float>((float *) &matB, 16);
        acc2 = fpmac(acc2, matB, 8, 0x76543210, matA, 3, 0x00000000);
        // printf("\n");
      }
      acc1 = fpmax(acc1, zeros, 0, 0x76543210);
      acc2 = fpmax(acc2, zeros, 0, 0x76543210);
      // printf("\n");
      // print_fvec<float>((float *) &acc1, 8);
      // print_fvec<float>((float *) &acc2, 8);
      // printf("\n");

      if (NCHUNK - j < 8) {
        float *acc_ptr = (float *) &acc1;
        for (int i = 0; i < NCHUNK - j; i++) {
          window_writeincr(out, acc_ptr[i]);
        }
      } else if (NCHUNK - j < 16) {
        window_writeincr(out, acc1);
        float *acc_ptr = (float *) &acc2;
        for (int i = 0; i < NCHUNK - j - 8; i++) {
          window_writeincr(out, acc_ptr[i]);
        }
      } else {
        window_writeincr(out, acc1);
        window_writeincr(out, acc2);
      }

      w_ptr += -K*NCHUNK_RND + 16;
      a_ptr -= K;
    }
    w_ptr -= NCHUNK_RND;
    a_ptr += K;
    
  }
  

  PROFILE_FOOTER;
}
