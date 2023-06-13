#include "conv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
void ConvReluScalarBHWC<INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU>::filter(
	input_window<float>* in,      // BHWC (1x28x28x1)
  output_window<float>* out     // BHWM (1x24x24x6)
) {
  PROFILE_HEADER(printf(
    "Running ConvReluScalarBHWC<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU));

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < OUT_H; h++) {
      for (int w = 0; w < OUT_W; w++) {
        
        for (int m = 0; m < M; m++) { 

          // KKC
          float res = bias[m];
          
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              for (int c = 0; c < C; c++) {
                res += window_readincr(in) * weights[weightIdx];
                weightIdx++;
              }
            }
            window_incr(in, C*(-K + INP_W)); // go back K, go down 1
          }

          if (IS_RELU)
            if (res < 0) res = 0;
          window_writeincr(out, res);
          window_incr(in, C*(-K*INP_W)); // go up K
        }
        weightIdx = 0;                   // reset weight
        window_incr(in, C);              // next position
      }
      window_incr(in, C*K - C);          // next row
    }
  }

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
void ConvReluScalarBCHW<INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU>::filter(
	input_window<float>* in,      // BCHW (1x1x28x28)
  output_window<float>* out     // BMHW (1x6x24x24)
) {
  PROFILE_HEADER(printf(
    "Running ConvReluScalarBCHW<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU));

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
        
          float res = bias[m];
          weightIdx = m*C*K*K;
          
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              for (int q = 0; q < K; q++) {
                float a = window_readincr(in);
                res += a * weights[weightIdx];
                weightIdx++;
              }
              window_incr(in, -K+INP_W); // go left K, down 1
            }
            window_incr(in, -K*INP_W + INP_H*INP_W); // go up K, channel 1
          }

          if (IS_RELU)
            if (res < 0) res = 0;
          window_writeincr(out, res);
          window_incr(in, -C*INP_H*INP_W + 1); // go channel -C, right 1
        }
        window_incr(in, INP_W-OUT_W); // go left OUT_W, go down 1
      }
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
    }
  }

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int K, int IS_RELU>
void ConvReluScalarBCHWStream<INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER(printf(
    "Running ConvReluScalarBCHWStream<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, K, IS_RELU));
  
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) {

      for (int i = 0; i < OUT_H*OUT_W; i++) {
        w_row[i] = bias[m];
      }
      
      for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
          for (int q = 0; q < K; q++) {

            float weight = readincr(weights);
            // printf("weight: %f\n", weight);
            
            for (int h = 0; h < OUT_H; h++) {
              for (int w = 0; w < OUT_W; w++) {
                float a = window_readincr(in);
                w_row[h*OUT_W + w] += weight * a;
                // printf("%f ", a);
              } // W
              window_incr(in, -OUT_W+INP_W); // next row after OUT_W
            } // H
            // printf("\n\n");
            
            window_incr(in, -OUT_H*INP_W+1);  // reset up OUT_H after OUT_W*OUT_W, go next
            
          } // K
          window_incr(in, -K+INP_W); // next row in KxK
          
        } // K
        window_incr(in, -K*INP_W);   // reset to 0, 0

      } // C

      for (int i = 0; i < OUT_H*OUT_W; i++) {
        float res = w_row[i];
        res = (res >= 0) ? res : 0;
        window_writeincr(out, res);
      }

    } // M
  } // B

  PROFILE_FOOTER;
}


#ifdef __X86SIM__
#define GET_WVEC(wp, zstart) \
  wvec = fpshuffle(*(v8float*) wp, 0, 0x00043210);
#else
#define GET_WVEC(wp, zstart) \
  wvec = fpshuffle(*(v8float*) wp, zstart, 0x00043210);
#endif

template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int _K_notused, int IS_RELU>
void Conv5x5ReluBCHW<INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, _K_notused, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER(printf(
    "Running Conv5x5ReluBCHW<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, _K_notused, IS_RELU));

  v16float data = null_v16float();
  v8float zeros = null_v8float();
  v8float wvec;

  float* wp;
  int zstart;

#define MAC_ROW(acc) \
  acc = fpmac(acc, data, 0, 0x76543210, wvec, 0, 0x00000000); \
  acc = fpmac(acc, data, 1, 0x76543210, wvec, 1, 0x00000000); \
  acc = fpmac(acc, data, 2, 0x76543210, wvec, 2, 0x00000000); \
  acc = fpmac(acc, data, 3, 0x76543210, wvec, 3, 0x00000000); \
  acc = fpmac(acc, data, 4, 0x76543210, wvec, 4, 0x00000000);

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { // computes one output channel
      for (int h = 0; h < OUT_W; h+=2) {
        for (int w = 0; w < OUT_W; w+=8) { // computes 8 output channel pixels
          
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          v8float acc2 = aie::broadcast<float, 8>(bias[m]);
          wp = weights + m*C*5*5;
          zstart = m*C*5*5 & 0x3;

          for (int c = 0; c < C; c++) { // computes 8 partial products over 5x5 kernel
            GET_WVEC(wp, zstart); wp += 5; zstart = (zstart + 1) & 0x3;
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc1);
            
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            
            GET_WVEC(wp, zstart); wp += 5; zstart = (zstart + 1) & 0x3;
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            
            GET_WVEC(wp, zstart); wp += 5; zstart = (zstart + 1) & 0x3;
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            
            GET_WVEC(wp, zstart); wp += 5; zstart = (zstart + 1) & 0x3;
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            
            GET_WVEC(wp, zstart); wp += 5; zstart = (zstart + 1) & 0x3;
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_H*INP_W - 5*INP_W - 16);
            MAC_ROW(acc2);
          }
          window_incr(in, -C*INP_H*INP_W + 8); // data go channel -C, right 8
                    
          if (IS_RELU) {
            acc1 = fpmax(acc1, zeros, 0, 0x76543210);
            acc2 = fpmax(acc2, zeros, 0, 0x76543210);
          }
          window_write(out, acc1);
          window_incr(out, OUT_W);
          window_write(out, acc2);
          window_incr(out, -OUT_W+8);

        } // W
        window_incr(in, 2*INP_W-OUT_W/8*8); // go left OUT_W/8*8, go down 1
        window_incr(out, OUT_W);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
    } // M
  } // B

#undef MAC_ROW

  PROFILE_FOOTER;
}


template <int INP_H, int INP_W, int OUT_W, int STEP_H, int STEP_W, 
          int B, int C, int M, int _K_notused, int IS_RELU>
void Conv5x5on8ReluBCHW<INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, _K_notused, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER(printf(
    "Running Conv5x5on8ReluBCHW<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>\n", 
    INP_H, INP_W, OUT_W, STEP_H, STEP_W, B, C, M, _K_notused, IS_RELU));

  v16float data = null_v16float();
  v8float zeros = null_v8float();
  v8float* wvec = (v8float *) weights;

#define MAC_ROW(acc) \
  acc = fpmac(acc, data, 0, 0x76543210, *wvec, 0, 0x00000000); \
  acc = fpmac(acc, data, 1, 0x76543210, *wvec, 1, 0x00000000); \
  acc = fpmac(acc, data, 2, 0x76543210, *wvec, 2, 0x00000000); \
  acc = fpmac(acc, data, 3, 0x76543210, *wvec, 3, 0x00000000); \
  acc = fpmac(acc, data, 4, 0x76543210, *wvec, 4, 0x00000000);

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { // computes one output channel
      for (int h = 0; h < OUT_H; h+=2) {
        for (int w = 0; w < OUT_W; w+=8) { // computes 8 output channel pixels
          
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          v8float acc2 = aie::broadcast<float, 8>(bias[m]);

          for (int c = 0; c < C; c++) { // computes 8 partial products over 5x5 kernel
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            wvec++;
            
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            wvec++;
            
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            wvec++;
            
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            MAC_ROW(acc2);
            wvec++;
            
            MAC_ROW(acc1);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_H*INP_W - 5*INP_W - 16);
            MAC_ROW(acc2);
            wvec++;
          }
          window_incr(in, -C*INP_H*INP_W + 8); // data go channel -C, right 8
                    
          if (IS_RELU) {
            acc1 = fpmax(acc1, zeros, 0, 0x76543210);
            acc2 = fpmax(acc2, zeros, 0, 0x76543210);
          }
          window_write(out, acc1);
          window_incr(out, OUT_W);
          window_write(out, acc2);
          window_incr(out, -OUT_W+8);
          wvec -= C*5;

        } // W
        window_incr(in, 2*INP_W-OUT_W); // go left OUT_W, go down 1
        window_incr(out, OUT_W);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
      wvec += C*5;
    } // M
  } // B

#undef MAC_ROW

  PROFILE_FOOTER;
}