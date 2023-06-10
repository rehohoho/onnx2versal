#include "gemm.h"
#include "kernel_utils.h"



template <int M, int K, int N, int IS_RELU>
void GemmReluScalarMKNKStream<M, K, N, IS_RELU>::filter(
	input_stream<float>* in,      // MxK
  input_stream<float>* weight,  // NxK 
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKNKStream<%d,%d,%d,%d>\n", M, K, N, IS_RELU));
  
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++)
      in_row[k] = readincr(in);
    
    for (int j = 0; j < N; j++) {
      float res = bias[j];
      for (int k = 0; k < K; k++) {
        float a = in_row[k];
        float b = readincr(weight);
        res += a * b; // matB is a circular buffer
      }

      if (IS_RELU)
        if (res < 0) res = 0;
      window_writeincr(out, res);
    }
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N, int IS_RELU>
void GemmReluMKKNStream<M, K, N, IS_RELU>::filter(
	input_stream<float>* in,      // MxK
  input_stream<float>* weight,  // KxN
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER(printf(
    "Running GemmReluMKKNStream<%d,%d,%d,%d>\n", M, K, N, IS_RELU));
  
  float *out_ptr = (float *) out->ptr;
  float *b_ptr = (float *) bias;
  v8float zeros = null_v8float();
  v8float matA = undef_v8float();
  v16float matW = null_v16float();

#define MAC_ROW(matA_i) \
  matW = upd_v(matW, 0, readincr_v4(weight)); \
  matW = upd_v(matW, 1, readincr_v4(weight)); \
  acc1 = fpmac(acc1, matW, 0, 0x76543210, matA, matA_i, 0x00000000); \
  matW = upd_v(matW, 2, readincr_v4(weight)); \
  matW = upd_v(matW, 3, readincr_v4(weight)); \
  acc2 = fpmac(acc2, matW, 8, 0x76543210, matA, matA_i, 0x00000000);

  for (int i = 0; i < M; i++) {
    // read stream iter
    v8float *in_ptr = (v8float *) in_row;
    v8float acc1 = *(v8float *) (b_ptr + 0);
    v8float acc2 = *(v8float *) (b_ptr + 8);

    for (int k = 0; k < K-7; k+=8) { // += input[k:k+8]*weight[k:k+8,n:n+16]
      matA = upd_v(matA, 0, readincr_v4(in));

      MAC_ROW(0); // += input[0]*weight[0:16]
      MAC_ROW(1);
      MAC_ROW(2);
      MAC_ROW(3);
      
      matA = upd_v(matA, 1, readincr_v4(in));
      *in_ptr = matA;
      MAC_ROW(4);
      MAC_ROW(5);
      MAC_ROW(6);
      MAC_ROW(7);
    } // K
    if (RUN_LASTCHUNK) { // K%4==0
      matA = upd_v(matA, 0, readincr_v4(in)); 
      *in_ptr = matA; // dangerous
      for (int p = 0; p < 4; p++) {
        MAC_ROW(p);
      }
    } // K
    
    if (IS_RELU) {
      acc1 = fpmax(acc1, zeros, 0, 0x76543210);
      acc2 = fpmax(acc2, zeros, 0, 0x76543210);
    }

    window_writeincr(out, acc1);
    window_writeincr(out, acc2);

    // rest of iters
    for (int j = 16; j < N; j+=16) { // v16float output per iter

      v8float acc1 = *(v8float *) (b_ptr + j);
      v8float acc2 = *(v8float *) (b_ptr + j + 8);
      float *in_row_ptr = (float *) in_row;

      for (int k = 0; k < K-7; k+=8) { // += input[k:k+8]*weight[k:k+8,n:n+16]
        matA = *(v8float *) in_row_ptr; in_row_ptr += 8;
        MAC_ROW(0); // += input[0]*weight[0:16]
        MAC_ROW(1);
        MAC_ROW(2);
        MAC_ROW(3);
        MAC_ROW(4);
        MAC_ROW(5);
        MAC_ROW(6);
        MAC_ROW(7);
      } // K
      if (RUN_LASTCHUNK) {
        matA = *(v8float *) in_row_ptr; in_row_ptr += K_REM8;
        for (int p = 0; p < K_REM8; p++) {
          MAC_ROW(p);
        }
      } // K
      
      if (IS_RELU) {
        acc1 = fpmax(acc1, zeros, 0, 0x76543210);
        acc2 = fpmax(acc2, zeros, 0, 0x76543210);
      }

      window_writeincr(out, acc1);
      window_writeincr(out, acc2);
      
      v8float testp = *(v8float *) out_ptr;
      print_fvec<float>((float *) &testp, 8);

    } // N
    
  } // M

#undef MAC_ROW

  PROFILE_FOOTER;
}


template <int M, int K, int N, int IS_RELU>
void GemmReluScalarMKNK<M, K, N, IS_RELU>::filter(
	input_window<float>* in,      // MxK
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKNK<%d,%d,%d,%d>\n", M, K, N, IS_RELU));

  int weightIdx;

  for (int i = 0; i < M; i++) {
    weightIdx = 0;
    for (int j = 0; j < N; j++) {
      float res = bias[j];
      for (int k = 0; k < K; k++) {
        float a = window_readincr(in);
        float b = weights[weightIdx];
        weightIdx++;
        res += a * b;
      }
      
      if (IS_RELU)
        if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N, int IS_RELU>
void GemmReluScalarMKKN<M, K, N, IS_RELU>::filter(
	input_window<float>* in,      // MxK
                                // KxN
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER(printf(
    "Running GemmReluScalarMKKN<%d,%d,%d,%d>\n", M, K, N, IS_RELU));

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
      if (IS_RELU)
        if (res < 0) res = 0;
      window_writeincr(out, res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N, int IS_RELU>
void GemmReluMKKN<M, K, N, IS_RELU>::filter(
	input_window<float>* in,      // MxK
                                // KxN
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER(printf(
    "Running GemmReluMKKN<%d,%d,%d,%d>\n", M, K, N, IS_RELU));

  float *a_ptr = (float *) in->ptr;
  float *w_ptr = (float *) weights;
  float *b_ptr = (float *) bias;
  v8float zeros = null_v8float();
  v8float matA = undef_v8float();
  v16float matB = null_v16float();

#define MAC_ROW(matA_i) \
  matB = upd_w(matB, 0, *(v8float*) w_ptr); \
  acc1 = fpmac(acc1, matB, 0, 0x76543210, matA, matA_i, 0x00000000); \
  matB = upd_w(matB, 1, *(v8float*) (w_ptr + 8)); w_ptr += N; \
  acc2 = fpmac(acc2, matB, 8, 0x76543210, matA, matA_i, 0x00000000);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j+=16) { // v16float output per iter

      v8float acc1 = *(v8float *) (b_ptr + j);
      v8float acc2 = *(v8float *) (b_ptr + j + 8);

      for (int k = 0; k < K-7; k+=8) { // += input[k:k+8]*weight[k:16]
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
      }
      
      if (IS_RELU) {
        acc1 = fpmax(acc1, zeros, 0, 0x76543210);
        acc2 = fpmax(acc2, zeros, 0, 0x76543210);
      }

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
    w_ptr -= (N+15)/16*16;
    a_ptr += K;
    
  }

#undef MAC_ROW
  PROFILE_FOOTER;
}
