#include "conv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define CONV_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU);


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvReluScalarBCHW<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW (1x1x28x28)
  output_window<float>* out     // BMHW (1x6x24x24)
) {
  PROFILE_HEADER2;

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
          weightIdx -= C_PER_M*KH*KW;
        } // W

        window_incr(out, OUT_W_PAD - OUT_W);
        window_incr(in, -OUT_W*STEP_W + INP_W*STEP_H); // go left OUT_W*STEP_W, go down STEP_H
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H*STEP_H
      weightIdx += C_PER_M*KH*KW;
      if (m % (M/GROUP) == M/GROUP - 1) {
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
          chess_separator_scheduler();

        } // W
        window_incr(in, 2*INP_W-OUT_W_PAD); // go left OUT_W_PAD, go down 1
        window_incr(out, OUT_W_PAD);
        chess_separator_scheduler();
      } // H
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
      chess_separator_scheduler();
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
void ConvHx4ReluBCHW<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER2;

  v16float data = null_v16float();
  v8float zeros = null_v8float();
  float *w_ptr = (float *) weights;
  float *b_ptr = (float *) bias;

#define MAC_ROW(acc) \
  for (int i = 0; i < KW; i++) { \
    acc = fpmac(acc, data, i, 0x76543210, *(v8float *) w_ptr, i, 0); \
  }

#define UPD_DATA \
  data = upd_w(data, 0, window_readincr_v8(in)); \
  data = upd_v(data, 2, window_readincr_v4(in)); \
  window_incr(in, INP_W - 12);

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { // computes one output channel
      for (int h = 0; h < OUT_H; h+=2) chess_prepare_for_pipelining chess_loop_range(OUT_H/2,) {
        for (int w = 0; w < OUT_W_PAD; w+=8) chess_prepare_for_pipelining chess_loop_range(OUT_W_PAD/8,) { // computes 8 output channel pixels
          
          v8float acc1 = aie::broadcast<float, 8>(*b_ptr);
          v8float acc2 = aie::broadcast<float, 8>(*b_ptr);

          for (int c = 0; c < C; c++) chess_prepare_for_pipelining chess_loop_range(C,) { // computes 8 partial products over 5x5 kernel
            UPD_DATA
            MAC_ROW(acc1);
            UPD_DATA
            MAC_ROW(acc2);
            w_ptr += 4;

            MAC_ROW(acc1);            
            UPD_DATA
            MAC_ROW(acc2);
            w_ptr += 4;

            MAC_ROW(acc1);
            UPD_DATA;
            MAC_ROW(acc2);
            w_ptr += 4;
            window_incr(in, INP_H*INP_W - (KH+1)*INP_W);
          }
          window_incr(in, -C*INP_H*INP_W + 8); // data go channel -C, right 8
          w_ptr -= CKK_ROW_SIZE;
                    
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
      w_ptr += CKK_ROW_SIZE;
      b_ptr ++;
    } // M
  } // B

#undef UPD_DATA
#undef MAC_ROW

  CONV_PROFILE_FOOTER("ConvHx4ReluBCHW");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv1x1Relu<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  int weightIdx = 0;
  int width_r;
  if (STEP_W == 2) {
    width_r = OUT_W % 4 == 0 ? 4 : OUT_W % 4;
  } else {
    width_r = OUT_W % 8 == 0 ? 8 : OUT_W % 8;
  }
  int select_mask = (1 << width_r) - 1; // selects first width_r
  
  aie::vector<float, 8> data = null_v8float();
  v8float zeros = null_v8float();
  v16float res = null_v16float();

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD - 8/STEP_W; w+=8/STEP_W) {
        
          aie::accum<accfloat, 8> acc1;
          acc1.from_vector(aie::broadcast<float, 8>(bias[m]), 0);
          
          for (int c = 0; c < C; c++) {
            data = window_read_v8(in);
            window_incr(in, INP_H*INP_W);
            acc1 = aie::mac(acc1, data, weights[weightIdx]); weightIdx++;
          } // C
          window_incr(in, -C*INP_H*INP_W + 8);
          weightIdx -= C;

          if (IS_RELU) {
            acc1 = fpmax(acc1, zeros, 0, 0x76543210);
          }

          if (STEP_W == 2) {
            acc1 = fpshuffle(acc1, 0, 0x00006420);
            window_writeincr(out, ext_v(acc1, 0));
          } else {
            window_writeincr(out, acc1);
          }
        } // W

        // handle width boundary
        aie::accum<accfloat, 8> acc1;
        acc1.from_vector(aie::broadcast<float, 8>(bias[m]), 0);
        
        for (int c = 0; c < C; c++) {
          data = window_read_v8(in);
          window_incr(in, INP_H*INP_W);
          acc1 = aie::mac(acc1, data, weights[weightIdx]); weightIdx++;
        } // C
        window_incr(in, -C*INP_H*INP_W + 8);
        weightIdx -= C;

        if (IS_RELU) {
          acc1 = fpmax(acc1, zeros, 0, 0x76543210);
        }

        v16float res = null_v16float();
        if (STEP_W == 2) {
          acc1 = fpshuffle(acc1, 0, 0x00006420);
          res = upd_w(res, 1, acc1);
          res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
          acc1 = ext_w(res, 0);
          window_writeincr(out, ext_v(acc1, 0));
        } else {
          res = upd_w(res, 1, acc1);
          res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
          acc1 = ext_w(res, 0);
          window_writeincr(out, acc1);
        }

        window_incr(in, -OUT_W_PAD*STEP_W+INP_W*STEP_H); // go left OUT_W_PAD*STEP_W, go down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H * STEP_H
      weightIdx += C;
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv1x1Relu");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvReluScalarStream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  int weightIdx;

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < C_PER_M*KH*KW; i++) {
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
      if (m % (M/GROUP) == M/GROUP - 1) {
        window_incr(in, C_PER_M*INP_H*INP_W); // next C_PER_M channels
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("ConvReluScalarStream");
}


#define MAC_ROW(acc) \
  for (int i = 0; i < KW; i++) { \
    acc = fpmac(acc, data, i, 0x76543210, *(v8float *) w_ptr, i, 0); \
  }

#define UPD_DATA \
  data = upd_w(data, 0, window_readincr_v8(in)); \
  data = upd_v(data, 2, window_readincr_v4(in)); \
  window_incr(in, INP_W - 12);

// double acc require store in cache and write where VLIW underutilized
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvHx4ReluStream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  float* w_ptr = (float *) ckk_row;
  int width_r;
  if (STEP_W == 2) {
    width_r = OUT_W % 4 == 0 ? 4 : OUT_W % 4;
  } else {
    width_r = OUT_W % 8 == 0 ? 8 : OUT_W % 8;
  }
  int select_mask = (1 << width_r) - 1; // selects first width_r
  
  v16float data = null_v16float();
  v8float zeros = null_v8float();

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=4) {
        *(v4float *) w_ptr = readincr_v4(weights); w_ptr += 4;
      }
      w_ptr -= CKK_ROW_SIZE;
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD - 8/STEP_W; w+=8/STEP_W) {
        
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          
          for (int c = 0; c < C_PER_M; c++) {
            for (int p = 0; p < KH; p++) chess_flatten_loop {
              UPD_DATA;
              MAC_ROW(acc1);
              w_ptr += 4;
            }
            window_incr(in, INP_H*INP_W - KH*INP_W);
          } // C_PER_M
          window_incr(in, -C_PER_M*INP_H*INP_W + 8);
          w_ptr -= CKK_ROW_SIZE;

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

        v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          
        for (int c = 0; c < C_PER_M; c++) {
          for (int p = 0; p < KH; p++) chess_flatten_loop {
            UPD_DATA;
            MAC_ROW(acc1);
            w_ptr += 4;
          }
          window_incr(in, INP_H*INP_W - KH*INP_W);
        } // C_PER_M
        window_incr(in, -C_PER_M*INP_H*INP_W + 8);
        w_ptr -= CKK_ROW_SIZE;

        if (IS_RELU) {
          acc1 = fpmax(acc1, zeros, 0, 0x76543210);
        }

        v16float res = null_v16float();
        if (STEP_W == 2) {
          acc1 = fpshuffle(acc1, 0, 0x00006420);
          res = upd_w(res, 1, acc1);
          res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
          acc1 = ext_w(res, 0);
          writeincr_v4(out, ext_v(acc1, 0));
        } else {
          res = upd_w(res, 1, acc1);
          res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
          acc1 = ext_w(res, 0);
          writeincr_v4(out, ext_v(acc1, 0));
          writeincr_v4(out, ext_v(acc1, 1));
        }

        window_incr(in, -OUT_W_PAD*STEP_W+INP_W*STEP_H); // go left OUT_W_PAD*STEP_W, go down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H * STEP_H
      if (m % (M/GROUP) == M/GROUP - 1) {
        window_incr(in, C_PER_M*INP_H*INP_W);
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("ConvHx4ReluStream");
}


// stride > 1 have no reuse of data down the row
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void ConvHx4ReluStreamMultiRow<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  float* w_ptr = (float *) ckk_row;
  
  v16float data = null_v16float();
  v8float zeros = null_v8float();

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=4) {
        *(v4float *) w_ptr = readincr_v4(weights); w_ptr += 4;
      }
      w_ptr -= CKK_ROW_SIZE;
      
      for (int h = 0; h < OUT_H; h+=2) {
        
        v8float *out_row_ptr = (v8float *) out_row;
        
        for (int w = 0; w < OUT_W_PAD; w+=8/STEP_W) {
        
          v8float acc1 = aie::broadcast<float, 8>(bias[m]);
          v8float acc2 = aie::broadcast<float, 8>(bias[m]);
          
          for (int c = 0; c < C; c++) {
            UPD_DATA;
            MAC_ROW(acc1);
            UPD_DATA;
            MAC_ROW(acc2);
            w_ptr += 4;

            MAC_ROW(acc1);
            UPD_DATA;
            MAC_ROW(acc2);
            w_ptr += 4;

            MAC_ROW(acc1);
            UPD_DATA;
            MAC_ROW(acc2);
            w_ptr += 4;
            
            window_incr(in, INP_H*INP_W - (KH+1)*INP_W);
          } // C
          window_incr(in, -C*INP_H*INP_W + 8);
          w_ptr -= CKK_ROW_SIZE;

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


  CONV_PROFILE_FOOTER("ConvHx4ReluStreamMultiRow");
}
#undef UPD_DATA
#undef MAC_ROW


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, 
          int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU>
void Conv1x1ReluStream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, IS_RELU>::filter(
	input_window<float>* in,      // BCHW
  input_stream<float>* weights, // MCKK
  output_stream<float>* out     // BMHW
) {
  PROFILE_HEADER2;
  
  float* w_ptr;
  int width_r;
  if (STEP_W == 2) {
    width_r = OUT_W % 4 == 0 ? 4 : OUT_W % 4;
  } else {
    width_r = OUT_W % 8 == 0 ? 8 : OUT_W % 8;
  }
  int select_mask = (1 << width_r) - 1; // selects first width_r
  
  aie::vector<float, 8> data = null_v8float();
  v8float zeros = null_v8float();
  v16float res = null_v16float();

  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      float* w_ptr = (float *) ckk_row;
      for (int i = 0; i < CKK_ROW_SIZE; i+=4) {
        *(v4float *) w_ptr = readincr_v4(weights); w_ptr += 4;
      }
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD - 8/STEP_W; w+=8/STEP_W) {
        
          aie::accum<accfloat, 8> acc1;
          acc1.from_vector(aie::broadcast<float, 8>(bias[m]), 0);
          
          for (int c = 0; c < C; c++) {
            data = window_read_v8(in);
            window_incr(in, INP_H*INP_W);
            acc1 = aie::mac(acc1, data, ckk_row[c]);
          } // C
          window_incr(in, -C*INP_H*INP_W + 8);

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

        // handle width boundary
        aie::accum<accfloat, 8> acc1;
          acc1.from_vector(aie::broadcast<float, 8>(bias[m]), 0);
        
        for (int c = 0; c < C; c++) {
          data = window_read_v8(in);
          window_incr(in, INP_H*INP_W);
          acc1 = aie::mac(acc1, data, ckk_row[c]);
        } // C
        window_incr(in, -C*INP_H*INP_W + 8);

        if (IS_RELU) {
          acc1 = fpmax(acc1, zeros, 0, 0x76543210);
        }

        v16float res = null_v16float();
        if (STEP_W == 2) {
          acc1 = fpshuffle(acc1, 0, 0x00006420);
          res = upd_w(res, 1, acc1);
          res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
          acc1 = ext_w(res, 0);
          writeincr_v4(out, ext_v(acc1, 0));
        } else {
          res = upd_w(res, 1, acc1);
          res = fpselect16(select_mask, res, 0, 0x76543210, 0x76543210, 0, 0xfedcba98, 0xfedcba98);
          acc1 = ext_w(res, 0);
          writeincr_v4(out, ext_v(acc1, 0));
          writeincr_v4(out, ext_v(acc1, 1));
        }

        window_incr(in, -OUT_W_PAD*STEP_W+INP_W*STEP_H); // go left OUT_W_PAD*STEP_W, go down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H * STEP_H
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("Conv1x1ReluStream");
}