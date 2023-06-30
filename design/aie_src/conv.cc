#include "conv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define CONV_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU);


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvReluScalarBHWC<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BHWC (1x28x28x1)
  output_window<float>* out     // BHWM (1x24x24x6)
) {
  PROFILE_HEADER2;

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
      for (int w = 0; w < OUT_W_PAD; w++) chess_prepare_for_pipelining chess_loop_range(OUT_W_PAD,) {
        
        for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) {

          // KKC
          float res = bias[m];
          
          for (int p = 0; p < KH; p++) chess_flatten_loop {
            for (int q = 0; q < KW; q++) chess_flatten_loop {
              for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) {
                res += window_readincr(in) * weights[weightIdx];
                weightIdx++;
              }
            }
            window_incr(in, C*(-KW + INP_W)); // go back KW, go down 1
          }

          if (IS_RELU) {
            res = std::max(res, 0.0f);
          }
          window_writeincr(out, res);
          window_incr(in, C*(-KH*INP_W)); // go up KH
        }
        weightIdx = 0;                   // reset weight
        window_incr(in, C);              // next position
      }
      window_incr(in, C*KW - C);         // next row
    }
  }

  CONV_PROFILE_FOOTER("ConvReluScalarBHWC");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvReluScalarBCHW<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW (1x1x28x28)
  output_window<float>* out     // BMHW (1x6x24x24)
) {
  PROFILE_HEADER2;

  // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1, C_PER_M, H, W)
  int C_PER_M = C / GROUP;
  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) chess_prepare_for_pipelining chess_loop_range(OUT_W_PAD,) {
        
          float res = bias[m];
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) {
            for (int p = 0; p < KH; p++) chess_flatten_loop {
              for (int q = 0; q < KW; q++) chess_flatten_loop {
                float a = window_readincr(in);
                res += a * weights[weightIdx];
                weightIdx++;
              }
              window_incr(in, -KW+INP_W); // go left KW, down 1
            }
            window_incr(in, -KH*INP_W + INP_H*INP_W); // go up KH, channel 1
          } // C

          if (IS_RELU) {
            res = std::max(res, 0.0f);
          }
          window_writeincr(out, res);
          window_incr(in, -C_PER_M*INP_H*INP_W + STEP_W); // go channel -C_PER_M, right STEP_W
          weightIdx -= C*KH*KW;
        } // W

        window_incr(out, OUT_W_PAD - OUT_W);
        window_incr(in, -OUT_W*STEP_W + INP_W*STEP_H); // go left OUT_W*STEP_W, go down STEP_H
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H*STEP_H
      weightIdx += C*KH*KW;
      if (m % C_PER_M == 0) {
        window_incr(in, C_PER_M*INP_H*INP_W); // next C_PER_M channels
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("ConvReluScalarBCHW");
}


#ifdef __X86SIM__
#define GET_WVEC(wp, zstart) \
  wvec = fpshuffle(*(v8float*) wp, 0, 0x00043210);
#else
#define GET_WVEC(wp, zstart) \
  wvec = fpshuffle(*(v8float*) wp, zstart, 0x00043210);
#endif

template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv5x5ReluBCHW<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER2;

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
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) {// computes one output channel
      for (int h = 0; h < OUT_W_PAD; h+=2) chess_prepare_for_pipelining chess_loop_range(OUT_H/2,) {
        for (int w = 0; w < OUT_W_PAD; w+=8) chess_prepare_for_pipelining chess_loop_range(OUT_W_PAD/8,) { // computes 8 output channel pixels
          
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          v8float acc2 = aie::broadcast<float, 8>(bias[m]);
          wp = weights + m*C*5*5;
          zstart = m*C*5*5 & 0x3;

          for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) { // computes 8 partial products over 5x5 kernel
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
          window_incr(out, OUT_W_PAD);
          window_write(out, acc2);
          window_incr(out, -OUT_W_PAD+8);

        } // W
        window_incr(in, 2*INP_W-OUT_W_PAD); // go left OUT_W_PAD, go down 1
        window_incr(out, OUT_W_PAD);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv5x5ReluBCHW");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv5x5on8ReluBCHW<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER2;

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
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) {// computes one output channel
      for (int h = 0; h < OUT_H; h+=2) chess_prepare_for_pipelining chess_loop_range(OUT_H/2,) {
        for (int w = 0; w < OUT_W_PAD; w+=8) chess_prepare_for_pipelining chess_loop_range(OUT_W_PAD/8,) { // computes 8 output channel pixels
          
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          v8float acc2 = aie::broadcast<float, 8>(bias[m]);

          for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) { // computes 8 partial products over 5x5 kernel
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
          window_incr(out, OUT_W_PAD);
          window_write(out, acc2);
          window_incr(out, -OUT_W_PAD+8);
          wvec -= C*5;

        } // W
        window_incr(in, 2*INP_W-OUT_W_PAD); // go left OUT_W_PAD, go down 1
        window_incr(out, OUT_W_PAD);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
      wvec += C*5;
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv5x5on8ReluBCHW");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv3x3on12ReluBCHW<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER2;

  v16float data = null_v16float();
  v8float zeros = null_v8float();
  float *w_ptr = (float *) weights;
  float *b_ptr = (float *) bias;

#define MAC_ROW(acc, w_i) \
  acc = fpmac(acc, data, 0, 0x76543210, *(v8float *) w_ptr, w_i+0, 0); \
  acc = fpmac(acc, data, 1, 0x76543210, *(v8float *) w_ptr, w_i+1, 0); \
  acc = fpmac(acc, data, 2, 0x76543210, *(v8float *) w_ptr, w_i+2, 0);

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { // computes one output channel
      for (int h = 0; h < OUT_H; h+=2) chess_prepare_for_pipelining chess_loop_range(OUT_H/2,) {
        for (int w = 0; w < OUT_W_PAD; w+=8) chess_prepare_for_pipelining chess_loop_range(OUT_W_PAD/8,) { // computes 8 output channel pixels
          
          v8float acc1 = aie::broadcast<float, 8>(*b_ptr);
          v8float acc2 = aie::broadcast<float, 8>(*b_ptr);

          for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) { // computes 8 partial products over 5x5 kernel
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc1, 0);
            
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc2, 0);
            MAC_ROW(acc1, 3);
            
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc2, 3);
            w_ptr += 4;
            MAC_ROW(acc1, 2);
            
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_H*INP_W - 3*INP_W - 12);
            MAC_ROW(acc2, 2);
            w_ptr += 8;
          }
          window_incr(in, -C*INP_H*INP_W + 8); // data go channel -C, right 8
          w_ptr -= C*12;
                    
          if (IS_RELU) {
            acc1 = fpmax(acc1, zeros, 0, 0x76543210);
            acc2 = fpmax(acc2, zeros, 0, 0x76543210);
          }
          window_write(out, acc1);
          window_incr(out, OUT_W_PAD);
          window_write(out, acc2);
          window_incr(out, -OUT_W_PAD+8);

        } // W
        window_incr(in, -OUT_W_PAD*STEP_W + 2*INP_W*STEP_H); // go left OUT_W_PAD*STEP_W, go down 2*STEP_H
        window_incr(out, OUT_W_PAD);
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H * STEP_H
      w_ptr += C*12;
      b_ptr ++;
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv3x3on12ReluBCHW");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvReluScalarStreamCacheCKK<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  // each m kernel of shape (1,C_PER_M,K,K) applied on input of shape (1, C_PER_M, H, W)
  int C_PER_M = C / GROUP;
  int weightIdx;

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < C/GROUP*KH*KW; i++) {
        ckk_row[i] = readincr(weights);
      }
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
        
          float res = bias[m];
          weightIdx = 0;
          
          for (int c = 0; c < C_PER_M; c++) {
            for (int p = 0; p < KH; p++) {
              for (int q = 0; q < KW; q++) {
                float a = window_readincr(in);
                res += a * ckk_row[weightIdx];
                weightIdx++;
              }
              window_incr(in, -KW+INP_W); // go left KW, down 1
            }
            window_incr(in, -KH*INP_W + INP_H*INP_W); // go up KH, channel 1
          } // C

          if (IS_RELU) {
            if (res < 0) res = 0;
          }
          writeincr(out, res);
          window_incr(in, -C_PER_M*INP_H*INP_W + STEP_W); // go channel -C_PER_M, right STEP_W
        } // W

        for (int w = 0; w < OUT_W_PAD - OUT_W; w++) {
          writeincr(out, 0);
        }
        window_incr(in, -OUT_W*STEP_W + INP_W*STEP_H); // go left OUT_W*STEP_W, go down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H*STEP_H
      if (m % C_PER_M == 0) {
        window_incr(in, C_PER_M*INP_H*INP_W); // next C_PER_M channels
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("ConvReluScalarStreamCacheCKK");
}


// double acc require store in cache and write where VLIW underutilized
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv3x3ReluStreamCacheCKK<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  float* w_ptr = (float *) ckk_row;
  
  v16float data = null_v16float();
  v8float zeros = null_v8float();

#define MAC_ROW(acc, w_i) \
  acc = fpmac(acc, data, 0, 0x76543210, *(v8float *) w_ptr, w_i+0, 0); \
  acc = fpmac(acc, data, 1, 0x76543210, *(v8float *) w_ptr, w_i+1, 0); \
  acc = fpmac(acc, data, 2, 0x76543210, *(v8float *) w_ptr, w_i+2, 0);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < C*12; i+=4) {
        *(v4float *) w_ptr = readincr_v4(weights); w_ptr += 4;
      }
      w_ptr -= C*12;
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=8/STEP_W) {
        
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          
          for (int c = 0; c < C; c++) {
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc1, 0);

            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc1, 3);
            w_ptr += 4;

            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_H*INP_W - 2*INP_W - 12);
            MAC_ROW(acc1, 2);
            w_ptr += 8;
          } // C
          window_incr(in, -C*INP_H*INP_W + 8);
          w_ptr -= C*12;

          if (IS_RELU) {
            acc1 = fpmax(acc1, zeros, 0, 0x76543210);
          }

          if (STEP_W == 2) {
            acc1 = fpshuffle(acc1, 0, 0x00006420);
            writeincr_v4(out, ext_v(acc1, 0));
          } else {
            writeincr_v4(out, ext_v(acc1, 0));
            writeincr_v4(out, ext_v(acc1, 1));
          }
        } // W
        window_incr(in, -OUT_W_PAD*STEP_W+INP_W*STEP_H); // go left OUT_W_PAD*STEP_W, go down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H * STEP_H
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv3x3ReluStreamCacheCKK");
}


// stride > 1 have no reuse of data down the row
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv3x3ReluStreamCacheCKKMultiRow<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  float* w_ptr = (float *) ckk_row;
  
  v16float data = null_v16float();
  v8float zeros = null_v8float();

#define MAC_ROW(acc, w_i) \
  acc = fpmac(acc, data, 0, 0x76543210, *(v8float *) w_ptr, w_i+0, 0); \
  acc = fpmac(acc, data, 1, 0x76543210, *(v8float *) w_ptr, w_i+1, 0); \
  acc = fpmac(acc, data, 2, 0x76543210, *(v8float *) w_ptr, w_i+2, 0);

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < C*12; i+=4) {
        *(v4float *) w_ptr = readincr_v4(weights); w_ptr += 4;
      }
      w_ptr -= C*12;
      
      for (int h = 0; h < OUT_H; h+=2) {
        
        v8float *out_row_ptr = (v8float *) out_row;
        
        for (int w = 0; w < OUT_W_PAD; w+=8/STEP_W) {
        
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          v8float acc2 = aie::broadcast<float, 8>(bias[m]);
          
          for (int c = 0; c < C; c++) {
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc1, 0);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc2, 0);

            MAC_ROW(acc1, 3);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_W - 12);
            MAC_ROW(acc2, 3);
            w_ptr += 4;

            MAC_ROW(acc1, 2);
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_v(data, 2, window_readincr_v4(in));
            window_incr(in, INP_H*INP_W - 3*INP_W - 12);
            MAC_ROW(acc2, 2);
            w_ptr += 8;
          } // C
          window_incr(in, -C*INP_H*INP_W + 8);
          w_ptr -= C*12;

          if (IS_RELU) {
            acc1 = fpmax(acc1, zeros, 0, 0x76543210);
            acc2 = fpmax(acc2, zeros, 0, 0x76543210);
          }

          writeincr_v4(out, ext_v(acc1, 0));
          writeincr_v4(out, ext_v(acc1, 1));
          *out_row_ptr = acc2; out_row_ptr++;
        } // W

        window_incr(in, -OUT_W_PAD*STEP_W+2*INP_W*STEP_H); // go left OUT_W_PAD*STEP_W, go down 2*STEP_H
        
        v4float *_out_row_ptr = (v4float *) out_row;
        for (int i = 0; i < OUT_W_PAD; i+=4) {
          writeincr_v4(out, *_out_row_ptr); _out_row_ptr++;
        }

        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency

      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H * STEP_H
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv3x3ReluStreamCacheCKKMultiRow");
}
