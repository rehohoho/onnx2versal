#include "qlinearconv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define CONV_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP);


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvScalar<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER2;

  int weightIdx;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
        
          int res = bias[m];
          weightIdx = m*C*KH*KW;
          
          for (int c = 0; c < C_PER_M; c++) {
            for (int p = 0; p < KH; p++) {
              for (int q = 0; q < KW; q++) {
                int a = window_readincr(in);
                res += a * (weights[weightIdx]-w_zero);
                weightIdx++;
              }
              window_incr(in, -KW+INP_W); // go left KW, down 1
            }
            window_incr(in, -KH*INP_W + INP_H*INP_W); // go up KH, channel 1
          }
          res = y_zero + round(scale * res);
          res = std::min(std::max(res, -128), 127);

          window_writeincr(out, (int8_t) res);
          window_incr(in, -C_PER_M*INP_H*INP_W + STEP_W); // go channel -C_PER_M, right STEP_W
        } // W

        for (int w = 0; w < OUT_W_PAD - OUT_W; w++)
          window_writeincr(out, y_zero);

        window_incr(in, -OUT_W*STEP_W + INP_W*STEP_H); // go left OUT_W*STEP_W, go down STEP_H
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H*STEP_H
      if (m % (M/GROUP) == M/GROUP - 1) {
        window_incr(in, C_PER_M*INP_H*INP_W); // next C_PER_M channels
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConvScalar");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv5x5<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv5x5(
  int8_t (&w)[M*C*KH*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  scalebits = std::abs(log(x_scale*w_scale/y_scale) / log(2)) + 15;
  assert(scalebits <= 27); // KH*KW*int8*int8*scale <= acc48, for KH=KW=5
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

/**
 * QLinearConv5x5<28,32,24,32,1,1,6,5>
 * 
 * https://docs.xilinx.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-on-8x8-bits
 * 24 selects 4*4=16, (4+2+1)*4=28 => rows (16,18),(17,19),(28,30),(29,30) before square
 * square executes on 4x2 matrix
 * 
 * int8 * int8:
 * Requires: 
 *  x indexing %4, z indexing %2
 *  expand 5 weights into 16 long vector [0,0,0,0, a,a, b,b, c,c, d,d, e,e, 0,0]
 * 
 * acc0  += x4*z0  + x6*z1   x8*z2  + x10*z3  x12*z4 
 * acc1  += x5*z1  + x7*z2   x9*z3  + x11*z4  x13*z5
 * acc2  += x0       x2      x4*z2  + x6*z3   x8*z4  + x10*z5   x12*z6
 * acc3  += x1       x3      x5*z3  + x7*z4   x9*z5  + x11*z6   x13*z7
 * 
 * acc4  += x4*z4  + x6*z5   x8*z6  + x10*z7  x12*z8
 * acc5  += x5*z5  + x7*z6   x9*z7  + x11*z8  x13*z9
 * acc6  +=                  x4*z6  + x6*z7   x8*z8  + x10*z9   x12*z10
 * acc7  +=                  x5*z7  + x7*z8   x9*z9  + x11*z10  x13*z11
 * 
 * acc8  += x4*z8  + x6*z9   x8*z10 + x10*z11 x12*z12
 * acc9  += x5*z9  + x7*z10  x9*z11 + x11*z12 x13*z13
 * acc10 +=                  x4*z10 + x6*z11  x8*z12 + x10*z13  x12*z14
 * acc11 +=                  x5*z11 + x7*z12  x9*z13 + x11*z14  x13*z15
 * 
 * acc12 += x4*z12 + x6*z13  x8*z14 + x10*z15 x12*z16
 * acc13 += x5*z13 + x7*z14  x9*z15 + x11*z16 x13*z17
 * acc14 +=                  x4*z14 + x6*z15  x8*z16 + x10*z17  x12*z18
 * acc15 +=                  x5*z15 + x7*z16  x9*z17 + x11*z18  x13*z19
 * 
 * Vector registers can hold 256 int8 at most, 128 int16 at most.
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv5x5<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v64int8 wvec = null_v64int8();
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  v16acc48 acc2 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  int8_t *in_ptr = (int8_t *) in->ptr;
  v16int8 *w_ptr = (v16int8 *) weights;
  v16int8 *out_ptr = (v16int8 *) out->ptr;

#define MAC_ROW(acc) \
  acc = mac16(acc, wvec, 0, 0x00000000, 4, 0x1032, data, 0, 0x76543210, 2, 0x2110);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h+=2) {
        for (int w = 0; w < OUT_W_PAD; w+=16) {

          acc1 = acc_bias;
          acc2 = acc_bias;
        
          for (int c = 0; c < C; c++) { // computes 2x16 partial products over 5x5 kernel
            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);
            
            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);

            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);

            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);

            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_H*INP_W - 5*INP_W; // channel +1, up 5
            MAC_ROW(acc2);
          }
          
          // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          acc2 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc2, 0), scale);
          *out_ptr = bsrs(acc1, scalebits);
          out_ptr += OUT_W_PAD/16;
          *out_ptr = bsrs(acc2, scalebits);
          out_ptr += 1-OUT_W_PAD/16;

          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
          w_ptr -= C*5;
        } // W
        
        in_ptr += 2*INP_W - OUT_W_PAD; // go left OUT_W_PAD, down 2
        out_ptr += OUT_W_PAD/16;
      } // H
      in_ptr -= OUT_H*INP_W; // go up OUT_H
      w_ptr += C*5;
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv5x5");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv5x5Scale32bit<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv5x5Scale32bit(
  int8_t (&w)[M*C*KH*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  scalebits = 31; // shift for float2fix in [-32:31]
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv5x5Scale32bit<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v64int8 wvec = null_v64int8();
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  v16acc48 acc2 = undef_v16acc48();
  aie::accum<acc80,8> acc_shift1;
  aie::accum<acc48,16> acc_bias;
  acc_shift1.from_vector(aie::broadcast<int16_t, 8>(y_zero), scalebits);

  int8_t *in_ptr = (int8_t *) in->ptr;
  v16int8 *w_ptr = (v16int8 *) weights;
  v16int8 *out_ptr = (v16int8 *) out->ptr;

#define MAC_ROW(acc) \
  acc = mac16(acc, wvec, 0, 0x00000000, 4, 0x1032, data, 0, 0x76543210, 2, 0x2110);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h+=2) {
        for (int w = 0; w < OUT_W_PAD; w+=16) {

          acc1 = acc_bias;
          acc2 = acc_bias;
        
          for (int c = 0; c < C; c++) { // computes 2x16 partial products over 5x5 kernel
            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);
            
            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);

            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);

            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc2);

            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            MAC_ROW(acc1);
            data = *(v32int8 *) in_ptr; in_ptr += INP_H*INP_W - 5*INP_W; // channel +1, up 5
            MAC_ROW(acc2);
          }
          
          v8int32 accbuf1_1 = lsrs(ext_lo(acc1), 0);
          auto aieacc1_1 = aie::mac(acc_shift1, (aie::vector<int32_t,8>) accbuf1_1, scale);
          v8int32 accbuf1_2 = lsrs(ext_hi(acc1), 0);
          auto aieacc1_2 = aie::mac(acc_shift1, (aie::vector<int32_t,8>) accbuf1_2, scale);
          auto aieacc1 = aie::concat(aieacc1_1, aieacc1_2);
          
          v8int32 accbuf2_1 = lsrs(ext_lo(acc2), 0);
          auto aieacc2_1 = aie::mac(acc_shift1, (aie::vector<int32_t,8>) accbuf2_1, scale);
          v8int32 accbuf2_2 = lsrs(ext_hi(acc2), 0);
          auto aieacc2_2 = aie::mac(acc_shift1, (aie::vector<int32_t,8>) accbuf2_2, scale);
          auto aieacc2 = aie::concat(aieacc2_1, aieacc2_2);
          
          *out_ptr = aieacc1.to_vector<int8_t>(scalebits);
          out_ptr += OUT_W_PAD/16;
          *out_ptr = aieacc2.to_vector<int8_t>(scalebits);
          out_ptr += 1-OUT_W_PAD/16;

          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
          w_ptr -= C*5;
        } // W
        
        in_ptr += 2*INP_W - OUT_W_PAD; // go left OUT_W_PAD, down 2
        out_ptr += OUT_W_PAD/16;
      } // H
      in_ptr -= OUT_H*INP_W; // go up OUT_H
      w_ptr += C*5;
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv5x5Scale32bit");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv3x3<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv3x3(
  int8_t (&w)[M*C*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = std::abs(log(x_scale*w_scale/y_scale) / log(2)) + 15;
  assert(scalebits <= 27); // KH*KW*int8*int8*scale <= acc48, for KH=5
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

/**
 * QLinearConv3x3<28,32,24,32,1,1,6,5>
 * 
 * int16 * int8:
 * expands 9 weights into 16 long vector [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0]
 * 
 * acc0 += z0*x0 + z1*x1 z2*x2 + z3*x3
 * acc1 += z0*x1 + z1*x2 z2*x3 + z3*x4
 * ...
 * acc14 += z0*x14 + z1*x15 z2*x16 + z3*x17
 * acc15 += z0*x15 + z1*x16 z2*x17 + z3*x18
 * 
 * Vector registers can hold 256 int8 at most, 128 int16 at most.
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv3x3<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v32int16 data = null_v32int16();
  v32int8 wvec = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  int8_t *in_ptr = (int8_t *) in->ptr;
  v16int8 *w_ptr = (v16int8 *) weights;

  int res_updi = 0;
  v16int16 res = null_v16int16(); // for STEP_W == 2
  
// xoffsets: 4b offset for lane 0,2,4,6, for 04, off0=2*4, off2=(0+4 +1)*2 => 8,9, 10,11
// xoffsetshi: 4b offset for lane 8,10,12,14, same selection scheme
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, data, 0, 0x03020100, 0x07060504, 2, 0x2110, wvec, widx, 0x0, 0x0, 2, 0x1010);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          acc1 = acc_bias;
        
          for (int c = 0; c < C; c++) { // computes 2x16 partial products over 3x3 kernel
            wvec = upd_v(wvec, 0, *w_ptr); w_ptr++;
            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 0);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 4);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_H*INP_W - 2*INP_W; // channel+1, up 2
            MAC_ROW(acc1, 8);
          }
          
          // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            if (res_updi == 1) {
              res = upd_v(res, 1, tmphalf);
              window_writeincr(out, pack(res));
            } else {
              res = upd_v(res, 0, tmphalf);
            }
            res_updi = (res_updi + 1) & 0x1;
          
          } else {
            window_writeincr(out, bsrs(acc1, scalebits));
          }

          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
          w_ptr -= C;
        } // W
        
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += C;
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv3x3");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvScalarStream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  input_stream<int8_t>* weights,
  output_stream<int8_t>* out
) {
  PROFILE_HEADER2;

  int weightIdx;
  v16int8 *ckk_row_ptr;
  
  int resvi = 0;
  v16int16 resv = null_v16int16();

#define WRITE_OUT(res) \
  resv = upd_elem(resv, resvi, res); \
  if (resvi == 15) writeincr_v16(out, pack(resv)); \
  resvi = (resvi + 1) & 0xf;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      ckk_row_ptr = (v16int8 *) ckk_row;
      for (int i = 0; i < CKK_ROW_SIZE; i+=16) {
        *ckk_row_ptr = aie::sub((aie::vector<int8_t,16>) readincr_v16(weights), w_zero); ckk_row_ptr++;
      }
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
        
          int res = bias[m];
          weightIdx = 0;
          
          for (int c = 0; c < C_PER_M; c++) {
            for (int p = 0; p < KH; p++) {
              for (int q = 0; q < KW; q++) {
                int a = window_readincr(in);
                res += a * ckk_row[weightIdx];
                weightIdx++;
              }
              window_incr(in, -KW+INP_W); // go left KW, down 1
            }
            window_incr(in, -KH*INP_W + INP_H*INP_W); // go up KH, channel 1
            weightIdx += 16 - KH*KW;
          }
          res = y_zero + round(scale * res);
          res = std::min(std::max(res, -128), 127);

          WRITE_OUT(res);
          window_incr(in, -C_PER_M*INP_H*INP_W + STEP_W); // go channel -C_PER_M, right STEP_W
        } // W

        for (int w = 0; w < OUT_W_PAD - OUT_W; w++) {
          WRITE_OUT(y_zero);
        }
        
        window_incr(in, -OUT_W*STEP_W + INP_W*STEP_H); // go left OUT_W*STEP_W, go down STEP_H
      } // H
      window_incr(in, -INP_W*OUT_H*STEP_H); // go up OUT_H*STEP_H
      if (m % (M/GROUP) == M/GROUP - 1) {
        window_incr(in, C_PER_M*INP_H*INP_W); // next C_PER_M channels
      }
    } // M
  } // B
#undef WRITE_OUT

  CONV_PROFILE_FOOTER("QLinearConvScalarStream");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv3x3Stream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv3x3Stream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 28); // KH*KW*int8*int8*scale <= acc48, for KH=KW=3
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

/**
 * QLinearConv3x3Stream<28,32,24,32,1,1,6,5>
 * 
 * int16 * int8:
 * expands 9 weights into 16 long vector [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0]
 * 
 * acc0 += z0*x0 + z1*x1 z2*x2 + z3*x3
 * acc1 += z0*x1 + z1*x2 z2*x3 + z3*x4
 * ...
 * acc14 += z0*x14 + z1*x15 z2*x16 + z3*x17
 * acc15 += z0*x15 + z1*x16 z2*x17 + z3*x18
 * 
 * Vector registers can hold 256 int8 at most, 128 int16 at most.
 */
template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv3x3Stream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  input_stream<int8_t>* weights,
  output_stream<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v32int16 data = null_v32int16();
  v32int8 wvec = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16int8 *ckk_row_ptr = (v16int8 *) ckk_row;;
  int8_t *in_ptr = (int8_t *) in->ptr;

  int res_updi = 0;
  v16int16 res = null_v16int16(); // for STEP_W == 2
  
// xoffsets: 4b offset for lane 0,2,4,6, for 04, off0=2*4, off2=(0+4 +1)*2 => 8,9, 10,11
// xoffsetshi: 4b offset for lane 8,10,12,14, same selection scheme
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, data, 0, 0x03020100, 0x07060504, 2, 0x2110, wvec, widx, 0x0, 0x0, 2, 0x1010);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          acc1 = acc_bias;
          
          for (int c = 0; c < C_PER_M; c++) { // computes 2x16 partial products over 3x3 kernel
            wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 0);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 4);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_H*INP_W - 2*INP_W; // channel+1, up 2
            MAC_ROW(acc1, 8);
          }
          ckk_row_ptr -= C_PER_M;
          in_ptr += 16 - C_PER_M*INP_H*INP_W; // go channel -C_PER_M, right 16
          
          // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            if (res_updi == 1) {
              res = upd_v(res, 1, tmphalf);
              writeincr_v16(out, pack(res));
            } else {
              res = upd_v(res, 0, tmphalf);
            }
            res_updi = (res_updi + 1) & 0x1;
          
          } else {
            writeincr_v16(out, bsrs(acc1, scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_W - OUT_W_PAD*STEP_H; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if (m % (M/GROUP) == M/GROUP - 1) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv3x3Stream");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv3x3StreamPad<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv3x3StreamPad(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 28); // KH*KW*int8*int8*scale <= acc48, for KH=KW=3
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv3x3StreamPad<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  input_stream<int8_t>* weights,
  output_stream<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v32int16 data = null_v32int16();
  v32int8 wvec = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16int8 *ckk_row_ptr = (v16int8 *) ckk_row;
  int8_t *in_ptr = (int8_t *) in->ptr;

  int res_updi = 0;
  v16int16 res = null_v16int16(); // for STEP_W == 2
  int width_r = OUT_W % 16 == 0 ? 16 : OUT_W % 16;
  int select_mask = (1 << width_r) - 1; // (1 << 0) - 1 selects all zeros
  
// xoffsets: 4b offset for lane 0,2,4,6, for 04, off0=2*4, off2=(0+4 +1)*2 => 8,9, 10,11
// xoffsetshi: 4b offset for lane 8,10,12,14, same selection scheme
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, data, 0, 0x03020100, 0x07060504, 2, 0x2110, wvec, widx, 0x0, 0x0, 2, 0x1010);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD-16/STEP_W; w+=16/STEP_W) {

          acc1 = acc_bias;
          
          for (int c = 0; c < C_PER_M; c++) { // computes 2x16 partial products over 3x3 kernel
            wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 0);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 4);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_H*INP_W - 2*INP_W; // channel+1, up 2
            MAC_ROW(acc1, 8);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel -C_PER_M, right 16
          ckk_row_ptr -= C_PER_M;
          
          // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            if (res_updi == 1) {
              res = upd_v(res, 1, tmphalf);
              writeincr_v16(out, pack(res));
            } else {
              res = upd_v(res, 0, tmphalf);
            }
            res_updi = (res_updi + 1) & 0x1;
          
          } else {
            writeincr_v16(out, bsrs(acc1, scalebits));
          }
        } // W

        // handle width boundary
        acc1 = acc_bias;
        
        for (int c = 0; c < C_PER_M; c++) { // computes 2x16 partial products over 3x3 kernel
          wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
          data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
          MAC_ROW(acc1, 0);

          data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
          MAC_ROW(acc1, 4);

          data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_H*INP_W - 2*INP_W; // channel+1, up 2
          MAC_ROW(acc1, 8);
        }
        in_ptr += -C_PER_M*INP_H*INP_W + 16;
        ckk_row_ptr -= C_PER_M;
        
        // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
        acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
        v16int16 tmp = srs(acc1, scalebits);
        if (STEP_W == 2) {
          v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
          if (res_updi == 1) {
            res = upd_v(res, 1, tmphalf);
            v32int16 fat_res = null_v32int16();
            fat_res = upd_w(fat_res, 0, res);
            fat_res = select32(select_mask, aie::broadcast<int16_t, 32>(y_zero), fat_res);
            writeincr_v16(out, pack(ext_w(fat_res, 0)));
          } else {
            res = upd_v(res, 0, tmphalf);
          }
          res_updi = (res_updi + 1) & 0x1;
        
        } else {
          v32int16 fat_res = null_v32int16();
          fat_res = upd_w(fat_res, 0, tmp);
          fat_res = select32(select_mask, aie::broadcast<int16_t, 32>(y_zero), fat_res);
          writeincr_v16(out, pack(ext_w(fat_res, 0)));
        }
        
        in_ptr += -OUT_W_PAD*STEP_W + INP_W*STEP_H; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv3x3StreamPad");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv3x3StreamScale32bit<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv3x3StreamScale32bit(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  scalebits = 31;
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv3x3StreamScale32bit<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  input_stream<int8_t>* weights,
  output_stream<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v32int16 data = null_v32int16();
  v32int8 wvec = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc80,8> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 8>(y_zero), scalebits);

  v16int8 *ckk_row_ptr;
  int8_t *in_ptr = (int8_t *) in->ptr;

  int res_updi = 0;
  v16int16 res = null_v16int16(); // for STEP_W == 2
  
// xoffsets: 4b offset for lane 0,2,4,6, for 04, off0=2*4, off2=(0+4 +1)*2 => 8,9, 10,11
// xoffsetshi: 4b offset for lane 8,10,12,14, same selection scheme
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, data, 0, 0x03020100, 0x07060504, 2, 0x2110, wvec, widx, 0x0, 0x0, 2, 0x1010);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      ckk_row_ptr = (v16int8 *) ckk_row;
      for (int i = 0; i < CKK_ROW_SIZE; i+=16) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          acc1 = acc_bias;
          ckk_row_ptr = (v16int8 *) ckk_row;
          
          for (int c = 0; c < C; c++) { // computes 2x16 partial products over 3x3 kernel
            wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 0);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_W;
            MAC_ROW(acc1, 4);

            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_H*INP_W - 2*INP_W; // channel+1, up 2
            MAC_ROW(acc1, 8);
          }
          
          // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
          v8int32 accbuf1_1 = lsrs(ext_lo(acc1), 0);
          auto aieacc1_1 = aie::mac(acc_shift, (aie::vector<int32_t,8>) accbuf1_1, scale);
          v8int32 accbuf1_2 = lsrs(ext_hi(acc1), 0);
          auto aieacc1_2 = aie::mac(acc_shift, (aie::vector<int32_t,8>) accbuf1_2, scale);
          auto fat_acc1 = aie::concat(aieacc1_1, aieacc1_2);

          if (STEP_W == 2) {
            auto tmp = fat_acc1.to_vector<int16_t>(scalebits);
            v8int16 tmphalf = aie::filter_even(tmp, 1);
            if (res_updi == 1) {
              res = upd_v(res, 1, tmphalf);
              writeincr_v16(out, pack(res));
            } else {
              res = upd_v(res, 0, tmphalf);
            }
            res_updi = (res_updi + 1) & 0x1;
          
          } else {
            writeincr_v16(out, fat_acc1.to_vector<int8_t>(scalebits));
          }
          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
        } // W
        
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv3x3StreamScale32bit");
}


template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv1x1Stream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv1x1Stream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 30); // KH*KW*int8*int8*scale <= acc48, for KH=KW=1
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv1x1Stream<INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<int8_t>* in,
  input_stream<int8_t>* weights,
  output_stream<int8_t>* out
) {
  PROFILE_HEADER2;
  
  v16int8 data = undef_v16int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16int8 *ckk_row_ptr;
  int8_t *in_ptr = (int8_t *) in->ptr;

  int res_updi = 0;
  v16int16 res = null_v16int16(); // for STEP_W == 2
  int width_r = OUT_W % 16 == 0 ? 16 : OUT_W % 16;
  int select_mask = (1 << width_r) - 1; // (1 << 0) - 1 selects all zeros
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      ckk_row_ptr = (v16int8 *) ckk_row;
      for (int i = 0; i < CKK_ROW_SIZE; i+=16) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD - 16/STEP_W; w+=16/STEP_W) {

          acc1 = acc_bias;
          ckk_row_ptr = (v16int8 *) ckk_row;
          
          for (int c = 0; c < C; c++) { // computes 2x16 partial products over 3x3 kernel
            data = *(v16int8 *) in_ptr; in_ptr += INP_H*INP_W; // channel+1
            acc1 = aie::mac((aie::accum<acc48,16>) acc1, (aie::vector<int8_t,16>) data, ckk_row[c]);
          }
          
          // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            if (res_updi == 1) {
              res = upd_v(res, 1, tmphalf);
              writeincr_v16(out, pack(res));
            } else {
              res = upd_v(res, 0, tmphalf);
            }
            res_updi = (res_updi + 1) & 0x1;
          
          } else {
            writeincr_v16(out, bsrs(acc1, scalebits));
          }
          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
        } // W
        
        // handle width boundary
        acc1 = acc_bias;
        ckk_row_ptr = (v16int8 *) ckk_row;
        
        for (int c = 0; c < C; c++) { // computes 2x16 partial products over 3x3 kernel
          data = *(v16int8 *) in_ptr; in_ptr += INP_H*INP_W; // channel+1
          acc1 = aie::mac((aie::accum<acc48,16>) acc1, (aie::vector<int8_t,16>) data, ckk_row[c]);
        }
        
        // use aieapi to reduce vector register usage, no add/mul with scalar for intrinsics
        acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
        v16int16 tmp = srs(acc1, scalebits);
        if (STEP_W == 2) {
          v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
          if (res_updi == 1) {
            res = upd_v(res, 1, tmphalf);
            v32int16 fat_res = null_v32int16();
            fat_res = upd_w(fat_res, 0, res);
            fat_res = select32(select_mask, aie::broadcast<int16_t, 32>(y_zero), fat_res);
            writeincr_v16(out, pack(ext_w(fat_res, 0)));
          } else {
            res = upd_v(res, 0, tmphalf);
          }
          res_updi = (res_updi + 1) & 0x1;
        
        } else {
          v32int16 fat_res = null_v32int16();
          fat_res = upd_w(fat_res, 0, tmp);
          fat_res = select32(select_mask, aie::broadcast<int16_t, 32>(y_zero), fat_res);
          writeincr_v16(out, pack(ext_w(fat_res, 0)));
        }
        in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
        
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
    } // M
  } // B

#undef MAC_ROW

  CONV_PROFILE_FOOTER("QLinearConv1x1Stream");
}
