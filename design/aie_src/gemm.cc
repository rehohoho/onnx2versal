#include "gemm.h"
#include "kernel_utils.h"



template <int M, int K, int N>
void GemmReluScalarGmemParamMKNK<M, K, N>::filter(
	input_window<float>* in,      // MxK  (1x256)
  input_window<float>* weight,  // NxK  (120x256)
  input_window<float>* bias,    // N    (120)
  output_window<float>* out     // MxN  (1x120)
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarGmemParamMKNK<%d,%d,%d>\n", M, K, N));
  
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float res = window_readincr(bias);
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = window_readincr(weight);
        res += a * b; // matB is a circular buffer
      }

      if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next row
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N>
void GemmReluScalarMKNK<M, K, N>::filter(
	input_window<float>* in,      // MxK  (1x256)
  output_window<float>* out     // MxN  (1x120)
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKNK<%d,%d,%d>\n", M, K, N));

  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
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
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N>
void GemmReluScalarMKKN<M, K, N>::filter(
	input_window<float>* in,      // MxK  inputs
                                // KxN  weights
  output_window<float>* out     // MxN  outputs
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKKN<%d,%d,%d>\n", M, K, N));

  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float res = bias[j];
      weightIdx = j;
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = weights[weightIdx];
        weightIdx += N;
        res += a * b;
      }    
      if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N>
void GemmReluMKKN<M, K, N>::filter(
	input_window<float>* in,      // MxK  inputs
                                // KxN  weights
  output_window<float>* out     // MxN  outputs
) {
  PROFILE_HEADER(printf(
    "Running GemmReluMKKN<%d,%d,%d>\n", M, K, N));

  float *a_ptr = (float *) in->ptr;
  float *w_ptr = (float *) weights;
  float *b_ptr = (float *) bias;
  v8float zeros = null_v8float();
  v8float matA = undef_v8float();
  v16float matB = null_v16float();
  const int KREM = K%8;

#define MAC_ROW(matA_i) \
  matB = upd_w(matB, 0, *(v8float*) w_ptr); \
  acc1 = fpmac(acc1, matB, 0, 0x76543210, matA, matA_i, 0x00000000); \
  matB = upd_w(matB, 1, *(v8float*) (w_ptr + 8)); w_ptr += N; \
  acc2 = fpmac(acc2, matB, 8, 0x76543210, matA, matA_i, 0x00000000);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j+=16) { // v16float output per iter

      v8float acc1 = *(v8float *) (b_ptr + j);
      v8float acc2 = *(v8float *) (b_ptr + j + 8);
      int k = 0;

      for (k = 0; k < K-7; k+=8) { // += input[k:k+8]*weight[k:16]
        matA = *(v8float *) a_ptr; a_ptr += 8;
        MAC_ROW(0); // += input[0]*weight[0:16]
        MAC_ROW(1);
        MAC_ROW(2);
        MAC_ROW(3);
        MAC_ROW(4);
        MAC_ROW(5);
        MAC_ROW(6);
        MAC_ROW(7);
      }
      if (RUN_LASTCHUNK) {
        matA = *(v8float *) a_ptr; a_ptr += K_REM8;
        for (int p = 0; p < K_REM8; p++) {
          MAC_ROW(p);
        }
        k+=K_REM8;
      }
      acc1 = fpmax(acc1, zeros, 0, 0x76543210);
      acc2 = fpmax(acc2, zeros, 0, 0x76543210);

      if (N - j < 8) {
        float *acc_ptr = (float *) &acc1;
        for (int i = 0; i < N - j; i++) {
          window_writeincr(out, acc_ptr[i]);
        }
      } else if (N - j < 16) {
        window_writeincr(out, acc1);
        float *acc_ptr = (float *) &acc2;
        for (int i = 0; i < N - j - 8; i++) {
          window_writeincr(out, acc_ptr[i]);
        }
      } else {
        window_writeincr(out, acc1);
        window_writeincr(out, acc2);
      }

      w_ptr += -K*N + 16;
      a_ptr -= K;
    }
    w_ptr -= N;
    a_ptr += K;
    
  }

  PROFILE_FOOTER;
}
