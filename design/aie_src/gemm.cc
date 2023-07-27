#include "gemm.h"
#include "kernel_utils.h"


#define GEMM_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d>", \
    filter_name, M, K, N, IS_RELU);


template <int M, int K, int N, int IS_RELU>
void GemmReluScalarMKNKStream<M, K, N, IS_RELU>::filter(
	input_stream<float>* in,      // MxK
  input_stream<float>* weight,  // NxK 
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER2;
  
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

  GEMM_PROFILE_FOOTER("GemmReluScalarMKNKStream");
}


template <int M, int K, int N, int IS_RELU>
void GemmReluMKKNStream<M, K, N, IS_RELU>::filter(
	input_stream<float>* in,      // MxK
  input_stream<float>* weight,  // KxN
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER2;
  
  float *b_ptr = (float *) bias;
  v8float matW = null_v8float();

#define MAC_ROW(k1, k2, k3, k4) \
  matW = upd_v(matW, 0, readincr_v4(weight)); \
  matW = upd_v(matW, 1, readincr_v4(weight)); \
  acc1 = aie::mac(acc1, (aie::vector<float, 8>) matW, k1); \
  acc2 = aie::mac(acc2, (aie::vector<float, 8>) matW, k2); \
  acc3 = aie::mac(acc3, (aie::vector<float, 8>) matW, k3); \
  acc4 = aie::mac(acc4, (aie::vector<float, 8>) matW, k4);

  for (int i = 0; i < M-3; i+=4) {
    v4float *in_ptr = (v4float *) in_row;
    for (int k = 0; k < 4*K; k+=4) {
      *in_ptr = readincr_v4(in); in_ptr ++;
    }

    float *out_row_ptr = (float *) out_row;

    for (int j = 0; j < N; j+=8) { // 2x v8float output per iter

      aie::accum<accfloat,8> acc1 = *(v8float *) (b_ptr + j);
      aie::accum<accfloat,8> acc2 = *(v8float *) (b_ptr + j);
      aie::accum<accfloat,8> acc3 = *(v8float *) (b_ptr + j);
      aie::accum<accfloat,8> acc4 = *(v8float *) (b_ptr + j);

      for (int k = 0; k < K-7; k+=8) { // += input[k:k+8]*weight[k:k+8,n:n+16]
        MAC_ROW(in_row[k+0], in_row[k+K+0], in_row[k+2*K+0], in_row[k+3*K+0]); // += input[0]*weight[0:16]
        MAC_ROW(in_row[k+1], in_row[k+K+1], in_row[k+2*K+1], in_row[k+3*K+1]);
        MAC_ROW(in_row[k+2], in_row[k+K+2], in_row[k+2*K+2], in_row[k+3*K+2]);
        MAC_ROW(in_row[k+3], in_row[k+K+3], in_row[k+2*K+3], in_row[k+3*K+3]);
        MAC_ROW(in_row[k+4], in_row[k+K+4], in_row[k+2*K+4], in_row[k+3*K+4]);
        MAC_ROW(in_row[k+5], in_row[k+K+5], in_row[k+2*K+5], in_row[k+3*K+5]);
        MAC_ROW(in_row[k+6], in_row[k+K+6], in_row[k+2*K+6], in_row[k+3*K+6]);
        MAC_ROW(in_row[k+7], in_row[k+K+7], in_row[k+2*K+7], in_row[k+3*K+7]);
      } // K

      int k = K/8*8;
      if (K%8 != 0) {
        for (int p = 0; p < 4; p++) {
          MAC_ROW(in_row[k+p], in_row[k+K+p], in_row[k+2*K+p], in_row[k+3*K+p]);
        }
      } // K
      
      aie::vector<float, 8> accv1 = acc1.to_vector<float>();
      aie::vector<float, 8> accv2 = acc2.to_vector<float>();
      aie::vector<float, 8> accv3 = acc3.to_vector<float>();
      aie::vector<float, 8> accv4 = acc4.to_vector<float>();
      if (IS_RELU) {
        accv1 = aie::max(accv1, 0.0f);
        accv2 = aie::max(accv2, 0.0f);
        accv3 = aie::max(accv3, 0.0f);
        accv4 = aie::max(accv4, 0.0f);
      }

      window_writeincr(out, accv1);
      aie::store_v(out_row_ptr, accv2); out_row_ptr += N;
      aie::store_v(out_row_ptr, accv3); out_row_ptr += N;
      aie::store_v(out_row_ptr, accv4); out_row_ptr += -2*N + 8;
    } // N

    out_row_ptr = (float *) out_row;
    for (int j = 0; j < 3*N; j+=8) {
      window_writeincr(out, *(v8float *) out_row_ptr); out_row_ptr += 8;
    }
    
  } // M
#undef MAC_ROW

#define MAC_ROW(k1) \
  matW = upd_v(matW, 0, readincr_v4(weight)); \
  matW = upd_v(matW, 1, readincr_v4(weight)); \
  acc1 = aie::mac(acc1, (aie::vector<float, 8>) matW, k1);
  
  for (int i = 0; i < M%4; i++) {
    v4float *in_ptr = (v4float *) in_row;
    for (int k = 0; k < K; k+=4) {
      *in_ptr = readincr_v4(in); in_ptr ++;
    }

    for (int j = 0; j < N; j+=8) { // 2x v8float output per iter

      aie::accum<accfloat,8> acc1 = *(v8float *) (b_ptr + j);

      for (int k = 0; k < K-7; k+=8) { // += input[k:k+8]*weight[k:k+8,n:n+16]
        MAC_ROW(in_row[k+0]); // += input[0]*weight[0:16]
        MAC_ROW(in_row[k+1]);
        MAC_ROW(in_row[k+2]);
        MAC_ROW(in_row[k+3]);
        MAC_ROW(in_row[k+4]);
        MAC_ROW(in_row[k+5]);
        MAC_ROW(in_row[k+6]);
        MAC_ROW(in_row[k+7]);
      } // K

      int k = K/8*8;
      if (K%8 != 0) {
        for (int p = 0; p < 4; p++) {
          MAC_ROW(in_row[k+p]);
        }
      } // K
      
      aie::vector<float, 8> accv1 = acc1.to_vector<float>();
      if (IS_RELU) {
        accv1 = aie::max(accv1, 0.0f);
      }

      window_writeincr(out, accv1);
    } // N
  }
#undef MAC_ROW

  GEMM_PROFILE_FOOTER("GemmReluMKKNStream");
}


template <int M, int K, int N, int IS_RELU>
void GemmReluScalarMKNK<M, K, N, IS_RELU>::filter(
	input_window<float>* in,      // MxK
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER2;

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

  GEMM_PROFILE_FOOTER("GemmReluScalarMKNK");
}


template <int M, int K, int N, int IS_RELU>
void GemmReluScalarMKKN<M, K, N, IS_RELU>::filter(
	input_window<float>* in,      // MxK
                                // KxN
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER2;

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

  GEMM_PROFILE_FOOTER("GemmReluScalarMKKN");
}


template <int M, int K, int N, int IS_RELU>
void GemmReluMKKN<M, K, N, IS_RELU>::filter(
	input_window<float>* in,      // MxK
                                // KxN
  output_window<float>* out     // MxN
) {
  PROFILE_HEADER2;

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

  GEMM_PROFILE_FOOTER("GemmReluMKKN");
}
