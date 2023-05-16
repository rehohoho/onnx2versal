#include "conv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


template <int INP_W, int OUT_W, int B, int C, int M, int K>
void ConvReluScalarGmemParamBHWC<INP_W, OUT_W, B, C, M, K>::filter(
	input_window<float>* in,      // BHWC (1x28x28x1)
  input_window<float>* weight,  // MKKC (6x5x5x1)
  input_window<float>* bias,    // M    (6)
  output_window<float>* out     // BHWM (1x24x24x6)
) {
  PROFILE_HEADER(printf(
    "Running ConvReluScalarGmemParamBHWC<%d, %d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M, K));
  
  // BHWM
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < OUT_W; h++) {
      for (int w = 0; w < OUT_W; w++) {
        
        for (int m = 0; m < M; m++) { 

          // KKC
          float res = window_readincr(bias);
          
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              for (int c = 0; c < C; c++) {
                res += window_readincr(in)*window_readincr(weight);
              }
            }
            window_incr(in, C*(-K + INP_W)); // go back K, go down 1
          }

          if (res < 0) res = 0;
          window_writeincr(out, res);
          window_incr(in, C*(-K*INP_W)); // go up K
        }

        window_incr(in, C);            // next position
        window_incr(weight, -M*K*K*C); // reset weight
        window_incr(bias, -M);         // reset bias
      }
      window_incr(in, C*K - C); // next row
    }
  }

  PROFILE_FOOTER;
}


template <int INP_W, int OUT_W, int B, int C, int M, int K>
void ConvReluScalarBHWC<INP_W, OUT_W, B, C, M, K>::filter(
	input_window<float>* in,      // BHWC (1x28x28x1)
  output_window<float>* out     // BHWM (1x24x24x6)
) {
  PROFILE_HEADER(printf(
    "Running ConvReluScalarBHWC<%d, %d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M, K));

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < OUT_W; h++) {
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


template <int INP_W, int OUT_W, int B, int C, int M, int K>
void ConvReluScalarBCHW<INP_W, OUT_W, B, C, M, K>::filter(
	input_window<float>* in,      // BCHW (1x1x28x28)
  output_window<float>* out     // BMHW (1x6x24x24)
) {
  PROFILE_HEADER(printf(
    "Running ConvReluScalarBCHW<%d, %d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M, K));

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      for (int h = 0; h < OUT_W; h++) {
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
            window_incr(in, -K*INP_W + INP_W*INP_W); // go up K, channel 1
          }

          if (res < 0) res = 0;
          window_writeincr(out, res);
          window_incr(in, -C*INP_W*INP_W + 1); // go channel -C, right 1
        }
        window_incr(in, INP_W-OUT_W); // go left OUT_W, go down 1
      }
      window_incr(in, -OUT_W*INP_W); // go up OUT_W
    }
  }

  PROFILE_FOOTER;
}


#ifdef __X86SIM__
#define GET_WVEC(wp, zstart) \
  wvec = fpshuffle(*(v8float*) wp, 0, 0x00043210);
#else
#define GET_WVEC(wp, zstart) \
  wvec = fpshuffle(*(v8float*) wp, zstart, 0x00043210);
#endif

template <int INP_W, int OUT_W, int B, int C, int M, int _K_notused>
void Conv5x5ReluBCHW<INP_W, OUT_W, B, C, M, _K_notused>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER(printf(
    "Running Conv5x5ReluBCHW<%d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M));

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
            window_incr(in, INP_W*INP_W - 5*INP_W - 16);
            MAC_ROW(acc2);
          }
          window_incr(in, -C*INP_W*INP_W + 8); // data go channel -C, right 8
                    
          acc1 = fpmax(acc1, zeros, 0, 0x76543210);
          window_write(out, acc1);
          window_incr(out, OUT_W);
          acc2 = fpmax(acc2, zeros, 0, 0x76543210);
          window_write(out, acc2);
          window_incr(out, -OUT_W+8);

        } // W
        window_incr(in, 2*INP_W-OUT_W/8*8); // go left OUT_W/8*8, go down 1
        window_incr(out, OUT_W);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -OUT_W*INP_W); // go up OUT_W
    } // M
  } // B

#undef MAC_ROW

  PROFILE_FOOTER;
}


template <int INP_W, int OUT_W, int B, int C, int M, int _K_notused>
void Conv5x5on8ReluBCHW<INP_W, OUT_W, B, C, M, _K_notused>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER(printf(
    "Running Conv5x5on8ReluBCHW<%d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M));

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
      for (int h = 0; h < OUT_W; h+=2) {
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
            window_incr(in, INP_W*INP_W - 5*INP_W - 16);
            MAC_ROW(acc2);
            wvec++;
          }
          window_incr(in, -C*INP_W*INP_W + 8); // data go channel -C, right 8
                    
          acc1 = fpmax(acc1, zeros, 0, 0x76543210);
          window_write(out, acc1);
          window_incr(out, OUT_W);
          acc2 = fpmax(acc2, zeros, 0, 0x76543210);
          window_write(out, acc2);
          window_incr(out, -OUT_W+8);
          wvec -= C*5;

        } // W
        window_incr(in, 2*INP_W-OUT_W/8*8); // go left OUT_W/8*8, go down 1
        window_incr(out, OUT_W);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -OUT_W*INP_W); // go up OUT_W
      wvec += C*5;
    } // M
  } // B

#undef MAC_ROW

  PROFILE_FOOTER;
}