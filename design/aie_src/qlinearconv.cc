#include "qlinearconv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define CONV_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), typeid(TTPARAM).name(), INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP);

template<typename TT>
auto get_wss_tt(int idx) = delete;

template<>
auto get_wss_tt<int8_t>(int idx) {
  return getb_wss(0);
}

template<>
auto get_wss_tt<uint8_t>(int idx) {
  return getub_wss(0);
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvScalar<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  output_window<TT>* out
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

          window_writeincr(out, (TT) res);
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


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv5x5<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv5x5(
  TTPARAM (&w)[M*C*KH*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
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
template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv5x5<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;
  
  v64int8 wvec = null_v64int8();
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  v16acc48 acc2 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  TT *in_ptr = (TT *) in->ptr;
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


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv5x5Scale32bit<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv5x5Scale32bit(
  TTPARAM (&w)[M*C*KH*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  assert(w_zero == 0);
  scalebits = 31; // shift for float2fix in [-32:31]
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv5x5Scale32bit<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;
  
  v64int8 wvec = null_v64int8();
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  v16acc48 acc2 = undef_v16acc48();
  aie::accum<acc80,8> acc_shift1;
  aie::accum<acc48,16> acc_bias;
  acc_shift1.from_vector(aie::broadcast<int16_t, 8>(y_zero), scalebits);

  TT *in_ptr = (TT *) in->ptr;
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
          
          *out_ptr = aieacc1.to_vector<TT>(scalebits);
          out_ptr += OUT_W_PAD/16;
          *out_ptr = aieacc2.to_vector<TT>(scalebits);
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


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv3x3<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv3x3(
  TTPARAM (&w)[M*C*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
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


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv3x3<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;
  
  v32int16 wvec = null_v32int16();
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  TT *in_ptr = (TT *) in->ptr;
  v16int8 *w_ptr = (v16int8 *) weights;

#define MAC_ROW(acc, widx) \
  acc = mac16(acc, wvec, widx, 0x0, 0x0, 2, 0x1010, data, 0, MAC_ZOFFSET, 0x87766554, 2, MAC_ZSQUARE);
  
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
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc1, 0);

            data = *(v32int8 *) in_ptr; in_ptr += INP_W;
            MAC_ROW(acc1, 4);

            data = *(v32int8 *) in_ptr; in_ptr += INP_H*INP_W - 2*INP_W; // channel+1, up 2
            MAC_ROW(acc1, 8);
          }
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int8 tmp = bsrs(acc1, scalebits);
            int *tmpint = (int *) &tmp;
            window_writeincr(out, tmpint[0]);
            window_writeincr(out, tmpint[1]);
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


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvScalarStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  input_stream<TTPARAM>* restrict weights,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;

  int weightIdx;
  v16w *ckk_row_ptr = (v16w *) ckk_row;
  
  int resvi = 0;
  v16int16 resv = null_v16int16();

#define WRITE_OUT(res) \
  resv = upd_elem(resv, resvi, res); \
  if (resvi == 15) writeincr_v16(out, ((aie::vector<int16,16>) resv).pack<TT>()); \
  resvi = (resvi + 1) & 0xf;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE / 16;
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
        
          int res = bias[m];
          weightIdx = 0;
          
          for (int c = 0; c < C_PER_M; c++) {
            for (int p = 0; p < KH; p++) {
              for (int q = 0; q < KW; q++) {
                TT a = window_readincr(in); 
                res += a * (ckk_row[weightIdx] - w_zero);
                weightIdx++;
              }
              window_incr(in, -KW+INP_W); // go left KW, down 1
            }
            window_incr(in, -KH*INP_W + INP_H*INP_W); // go up KH, channel 1
            weightIdx += CKK_ROW_SIZE/C_PER_M - KH*KW;
          }
          res = y_zero + round(scale * res);
          if ((std::is_same<TT, int8_t>::value)) {
            res = std::min(std::max(res, -128), 127);
          } else {
            res = std::min(std::max(res, 0), 255);
          }

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


/**
 * QLinearConvHx4Stream<28,32,24,32,1,1,6,5>
 * 
 * int16 * int8:
 * expands 9 weights into 16 long vector [a,b,c,0, d,e,f,0, g,h,i,0, 0,0,0,0]
 * 
 * stride1
 * acc0 += x0*z0 + x1*z1 x2*z2 + x3*z3
 * acc1 += x0*z1 + x1*z2 x2*z3 + x3*z4
 * ...
 * acc14 += x0*z14 + x1*z15 x2*z16 + x3*z17
 * acc15 += x0*z15 + x1*z16 x2*z17 + x3*z18
 * 
 * stride2
 * acc0 += x0*z0 + x1*z1 x2*z2 + x3*z3
 * acc1 += x0*z2 + x1*z3 x2*z4 + x3*x5
 * ...
 * acc6 += x0*z12 + x1*z13 x2*z14 + x3*z15
 * acc7 += x0*z14 + x1*z15 x2*z16 + x3*z17
 * 
 * stride4
 * acc0 += x0*z0  + x1*z1  x2*z2 + x3*z3
 * acc1 += x0*z4  + x1*z5  x2*z6 + x3*x7
 * ...
 * acc6 += x0*z24 + x1*z25 x2*z26 + x3*z27
 * acc7 += x0*z28 + x1*z29 x2*z30 + x3*z31
 * 
 * xoffsets: 4b offset for every two lanes, e.g. 0 4 => 4*2=8, (0+4+1)*2=10 => 8,9, 10,11
 * zoffsets: 4b offset for every lane, e.g. offset=4, step=4 => 4*2=8 => 8,9, 14,15
 */
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, wvec, widx, 0x0, 0x0, 2, 0x1010, data, 0, MAC_ZOFFSET, 0x87766554, 2, MAC_ZSQUARE); \
  if (!(std::is_same<TTPARAM,int8_t>::value)) \
    acc = msc16(acc, wzero, 0, 0x0, 0x0, 2, 0x1010, data, 0, MAC_ZOFFSET, 0x87766554, 2, MAC_ZSQUARE);

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConvHx4Stream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConvHx4Stream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 28); // KH*KW*int8*int8*scale <= acc48, for KH=KW=3
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx4Stream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  input_stream<TTPARAM>* weights,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  for (int i = 0; i < KW; i++)
    wzero = upd_elem(wzero, i, w_zero);
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w *ckk_row_ptr = (v16w *) ckk_row;
  TT *in_ptr = (TT *) in->ptr;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) chess_prepare_for_pipelining chess_loop_range(CKK_ROW_SIZE/16,) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=W_LOOP_STEP) {

          acc1 = acc_bias;
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) { // computes 2x16 partial products over 3x3 kernel
            
            for (int p = 0; p <= KH-4; p+=4) chess_flatten_loop {
              wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 0);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 4);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 8);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 12);
            }
            
            if ((KH & 0x3) != 0) {
              wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
              for (int p = 0; p < (KH & 0x3); p++) {
                data = *(v32 *) in_ptr; in_ptr += INP_W;
                MAC_ROW(acc1, p*4);
              }
            }

            in_ptr += INP_H*INP_W -KH*INP_W; // channel+1, up KH
          }
          in_ptr += -C_PER_M*INP_H*INP_W + W_LOOP_IN_STEP; // go channel -C_PER_M, right W_LOOP_IN_STEP
          ckk_row_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W > 1) {
            v16 tmp = ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits);
            int *tmpint = (int *) &tmp;
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if (m % (M/GROUP) == M/GROUP - 1) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConvHx4Stream");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConvHx4StreamScale32bit<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConvHx4StreamScale32bit(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);
  scalebits = 31;
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx4StreamScale32bit<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  input_stream<TTPARAM>* weights,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;
  
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  for (int i = 0; i < KW; i++)
    wzero = upd_elem(wzero, i, w_zero);
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc80,8> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 8>(y_zero), scalebits);

  v16w *ckk_row_ptr = (v16w *) ckk_row;
  TT *in_ptr = (TT *) in->ptr;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) chess_prepare_for_pipelining chess_loop_range(CKK_ROW_SIZE/16,) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=W_LOOP_STEP) {

          acc1 = acc_bias;
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) { // computes 2x16 partial products over 3x3 kernel
            
            for (int p = 0; p <= KH-4; p+=4) chess_flatten_loop {
              wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
              data = *(v32int8 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 0);
              data = *(v32int8 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 4);
              data = *(v32int8 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 8);
              data = *(v32int8 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 12);
            }
            
            if ((KH & 0x3) != 0) {
              wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
              for (int p = 0; p < (KH & 0x3); p++) {
                data = *(v32 *) in_ptr; in_ptr += INP_W;
                MAC_ROW(acc1, p*4);
              }
            }
            
            in_ptr += INP_H*INP_W -KH*INP_W; // channel+1, up KH
          }
          in_ptr += -C_PER_M*INP_H*INP_W + W_LOOP_IN_STEP; // go channel -C_PER_M, right W_LOOP_IN_STEP
          ckk_row_ptr -= CKK_ROW_SIZE/16;
          
          v8int32 accbuf1_1 = lsrs(ext_lo(acc1), 0);
          auto aieacc1_1 = aie::mac(acc_shift, (aie::vector<int32_t,8>) accbuf1_1, scale);
          v8int32 accbuf1_2 = lsrs(ext_hi(acc1), 0);
          auto aieacc1_2 = aie::mac(acc_shift, (aie::vector<int32_t,8>) accbuf1_2, scale);
          auto fat_acc1 = aie::concat(aieacc1_1, aieacc1_2);

          if (STEP_W > 1) {
            v16 tmp = fat_acc1.to_vector<TT>(scalebits);
            int *tmpint = (int *) &tmp;
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, fat_acc1.to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B


  CONV_PROFILE_FOOTER("QLinearConvHx4StreamScale32bit");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConvHx4PktStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConvHx4PktStream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 28); // KH*KW*int8*int8*scale <= acc48, for KH=KW=3
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx4PktStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_pktstream* in_s,
  input_stream<TTPARAM>* weights,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;
  
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  for (int i = 0; i < KW; i++)
    wzero = upd_elem(wzero, i, w_zero);
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w *ckk_row_ptr = (v16w *) ckk_row;
  TT *in_ptr = (TT *) in;

  // fill window
  for (int bc = 0; bc < B*C; bc++) chess_prepare_for_pipelining chess_loop_range(B*C,) {
    get_ss(0); // discard header
    for (int hw = 0; hw < INP_H*INP_W; hw+=16) {
      *(v16 *) in_ptr = get_wss_tt<TT>(0); in_ptr+=16;
    }
  }
  in_ptr = (TT *) in;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) chess_prepare_for_pipelining chess_loop_range(CKK_ROW_SIZE/16,) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=W_LOOP_STEP) {

          acc1 = acc_bias;
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) { // computes 2x16 partial products over 3x3 kernel
            
            for (int p = 0; p <= KH-4; p+=4) chess_flatten_loop {
              wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 0);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 4);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 8);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 12);
            }
            
            if ((KH & 0x3) != 0) {
              wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
              for (int p = 0; p < (KH & 0x3); p++) {
                data = *(v32 *) in_ptr; in_ptr += INP_W;
                MAC_ROW(acc1, p*4);
              }
            }
            
            in_ptr += INP_H*INP_W -KH*INP_W; // channel+1, up KH
          }
          in_ptr += -C_PER_M*INP_H*INP_W + W_LOOP_IN_STEP; // go channel -C_PER_M, right W_LOOP_IN_STEP
          ckk_row_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W > 1) {
            v16 tmp = ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits);
            int *tmpint = (int *) &tmp;
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConvHx4PktStream");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx4_0<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
  input_window<TT>* in,
  output_stream<acc48>* cout
) {
  PROFILE_HEADER2;
  
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  for (int i = 0; i < KW; i++)
    wzero = upd_elem(wzero, i, w_zero);
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_bias;

  v16w *w_ptr = (v16w *) weights;
  TT *in_ptr = (TT *) in->ptr;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=W_LOOP_STEP) {

          acc1 = acc_bias;
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) { // computes 2x16 partial products over 3x3 kernel
            
            for (int p = 0; p <= KH-4; p+=4) chess_flatten_loop {
              wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 0);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 4);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 8);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 12);
            }
            
            if ((KH & 0x3) != 0) {
              wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
              for (int p = 0; p < (KH & 0x3); p++) {
                data = *(v32 *) in_ptr; in_ptr += INP_W;
                MAC_ROW(acc1, p*4);
              }
            }
            
            in_ptr += INP_H*INP_W -KH*INP_W; // channel+1, up KH
          }
          in_ptr += -C_PER_M*INP_H*INP_W + W_LOOP_IN_STEP; // go channel -C_PER_M, right W_LOOP_IN_STEP
          w_ptr -= CKK_ROW_SIZE/16;

          writeincr_v8(cout, ext_lo(acc1));
          writeincr_v8(cout, ext_hi(acc1));
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += CKK_ROW_SIZE/16;
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConvHx4_0");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx4_1<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
  input_window<TT>* in,
  input_stream<acc48>* cin,
  output_stream<acc48>* cout
) {
  PROFILE_HEADER2;
  
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  for (int i = 0; i < KW; i++)
    wzero = upd_elem(wzero, i, w_zero);
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();

  v16w *w_ptr = (v16w *) weights;
  TT *in_ptr = (TT *) in->ptr;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=W_LOOP_STEP) {

          aie::accum<acc48,8> _acc1 = readincr_v8(cin);
          aie::accum<acc48,8> _acc2 = readincr_v8(cin);
          acc1 = aie::concat(_acc1, _acc2);
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) { // computes 2x16 partial products over 3x3 kernel
            
            for (int p = 0; p <= KH-4; p+=4) chess_flatten_loop {
              wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 0);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 4);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 8);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 12);
            }
            
            if ((KH & 0x3) != 0) {
              wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
              for (int p = 0; p < (KH & 0x3); p++) {
                data = *(v32 *) in_ptr; in_ptr += INP_W;
                MAC_ROW(acc1, p*4);
              }
            }
            
            in_ptr += INP_H*INP_W -KH*INP_W; // channel+1, up KH
          }
          in_ptr += -C_PER_M*INP_H*INP_W + W_LOOP_IN_STEP; // go channel -C_PER_M, right W_LOOP_IN_STEP
          w_ptr -= CKK_ROW_SIZE/16;

          writeincr_v8(cout, ext_lo(acc1));
          writeincr_v8(cout, ext_hi(acc1));
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += CKK_ROW_SIZE/16;
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConvHx4_1");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConvHx4_2<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConvHx4_2(
  TTPARAM (&w)[M*CKK_ROW_SIZE],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  weights(w), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 28); // KH*KW*int8*int8*scale <= acc48, for KH=KW=3
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx4_2<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
  input_window<TT>* in,
  input_stream<acc48>* cin,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;
  
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  for (int i = 0; i < KW; i++)
    wzero = upd_elem(wzero, i, w_zero);
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w *w_ptr = (v16w *) weights;
  TT *in_ptr = (TT *) in->ptr;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

     for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=W_LOOP_STEP) {

          aie::accum<acc48,8> _acc1 = readincr_v8(cin);
          aie::accum<acc48,8> _acc2 = readincr_v8(cin);
          acc1 = aie::concat(_acc1, _acc2);
          
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) { // computes 2x16 partial products over 3x3 kernel
            
            for (int p = 0; p <= KH-4; p+=4) chess_flatten_loop {
              print_vec<short, short>((short *) &wvec, 16);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 0);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 4);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 8);
              data = *(v32 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1, 12);
            }
            
            if ((KH & 0x3) != 0) {
              wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
              for (int p = 0; p < (KH & 0x3); p++) {
                data = *(v32 *) in_ptr; in_ptr += INP_W;
                MAC_ROW(acc1, p*4);
              }
            }
            
            in_ptr += INP_H*INP_W -KH*INP_W; // channel+1, up KH
          }
          in_ptr += -C_PER_M*INP_H*INP_W + W_LOOP_IN_STEP; // go channel -C_PER_M, right W_LOOP_IN_STEP
          w_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W > 1) {
            v16 tmp = ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits);
            int *tmpint = (int *) &tmp;
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += CKK_ROW_SIZE/16;
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConvHx4_2");
}
#undef MAC_ROW


/**
 * QLinearConvHx6x8bitStream<28,32,24,32,1,1,6,5>
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
template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConvHx6x8bitStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConvHx6x8bitStream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
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

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConvHx6x8bitStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  input_stream<TTPARAM>* restrict weights,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  v64int8 wvec = null_v64int8();
  v32int8 data = null_v32int8();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16int8 *ckk_row_ptr = (v16int8 *) ckk_row;;
  TT *in_ptr = (TT *) in->ptr;

#define MAC_ROW(acc) \
  acc = mac16(acc, wvec, 0, 0x00000000, 4, 0x1032, data, 0, 0x76543210, 2, 0x2110);
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 
      
      for (int i = 0; i < CKK_ROW_SIZE; i+=16) chess_prepare_for_pipelining chess_loop_range(CKK_ROW_SIZE/16,) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;

      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          acc1 = acc_bias;
        
          for (int c = 0; c < C_PER_M; c++) chess_prepare_for_pipelining chess_loop_range(C_PER_M,) {
            
            for (int p = 0; p < KH-1; p++) {
              wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
              data = *(v32int8 *) in_ptr; in_ptr += INP_W;
              MAC_ROW(acc1);
            }
            wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
            data = *(v32int8 *) in_ptr; in_ptr += INP_H*INP_W -(KH-1)*INP_W; // channel+1, up KH
            MAC_ROW(acc1);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel -C_PER_M, right 16
          ckk_row_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          writeincr(out, bsrs(acc1, scalebits));
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

  CONV_PROFILE_FOOTER("QLinearConvHx6x8bitStream");
}

/**
 * bandwidth constrained (compute:loads = 1:1)
 *          z0  z1    z16=0  z17=0
 * acc0  += x0  x16
 * acc1  += x1  x17
 * ...
 * acc14 += x14 x30
 * acc15 += x15 x31
 * 
 * 
 * int16 * int8:
 * xoffsets: 4b offset for every two lanes, e.g. 0 4 => 4*2=8, (0+4+1)*2=10 => 8,9, 10,11
 * zoffsets: 2b offset for every lane, e.g. offset=80, step=2 => 0*2=0  => 0, 1,  2, 3,
 *                                                               8*2=16 => 16,17, 18,19,
 */

// only use 2/4 columns
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, wvec, widx, 0x0, 0x0, 16, 0x1010, data, 0, 0xb3a29180, 0xf7e6d5c4, 2, 0x3120); \
  if (!(std::is_same<TTPARAM,int8_t>::value)) \
    acc = msc16(acc, wzero, 0, 0x0, 0x0, 16, 0x3210, data, 0, 0xb3a29180, 0xf7e6d5c4, 2, 0x3120); 

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv1x1Stream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv1x1Stream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);

  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 30); // KH*KW*int8*int8*scale <= acc48, for KH=KW=1
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv1x1Stream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_window<TT>* in,
  input_stream<TTPARAM>* restrict weights,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;

  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  wzero = upd_v(wzero, 0, aie::broadcast<int16_t,8>(w_zero));
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w* restrict ckk_row_ptr = (v16w *) ckk_row;
  TT* restrict in_ptr = (TT *) in->ptr;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int i = 0; i < CKK_ROW_SIZE; i+=16) chess_prepare_for_pipelining chess_loop_range(CKK_ROW_SIZE/16,) {
        *ckk_row_ptr = readincr_v16(weights); ckk_row_ptr++;
      }
      ckk_row_ptr -= CKK_ROW_SIZE/16;
      
      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) chess_prepare_for_pipelining chess_loop_range(OUT_H,) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          acc1 = acc_bias;
          
          for (int c = 0; c <= C_PER_M-16; c+=16) {
            wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
            for (int i = 0; i < 16; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 16 != 0) {
            wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
            for (int i = 0; i <= C_PER_M-2; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 2 != 0) {
            data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
            MAC_ROW(acc1, LAST_C);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel-C_PER_M, right 16
          ckk_row_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            tmp = upd_v(tmp, 0, tmphalf);
            aie::vector<TT,16> tmpout = ((aie::vector<int16_t,16>) tmp).pack<TT>();
            int *tmpint = (int *) &(tmpout);
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConv1x1Stream");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv1x1PktStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv1x1PktStream(
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 30); // KH*KW*int8*int8*scale <= acc48, for KH=KW=1
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv1x1PktStream<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
	input_pktstream* in_s,
  input_stream<TTPARAM>* restrict weights,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  wzero = upd_v(wzero, 0, aie::broadcast<int16_t,8>(w_zero));
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w* restrict ckk_row_ptr = (v16w *) ckk_row;
  TT* restrict in_ptr = (TT *) in;
  
  // fill window
  for (int bc = 0; bc < B*C; bc++) {
    get_ss(0); // discard header
    for (int hw = 0; hw < INP_H*INP_W; hw+=16) {
      *(v16 *) in_ptr = get_wss_tt<TT>(0); in_ptr+=16;
    }
  }
  in_ptr = (TT *) in;
  
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
          
          for (int c = 0; c <= C_PER_M-16; c+=16) {
            wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
            for (int i = 0; i < 16; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 16 != 0) {
            wvec = upd_w(wvec, 0, unpack(*ckk_row_ptr)); ckk_row_ptr++;
            for (int i = 0; i <= C_PER_M-2; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 2 != 0) {
            data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
            v16 zeros = aie::zeros<TT,16>();
            data = upd_v(data, 1, zeros);
            MAC_ROW(acc1, LAST_C);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel-C_PER_M, right 16
          ckk_row_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            tmp = upd_v(tmp, 0, tmphalf);
            aie::vector<TT,16> tmpout = ((aie::vector<int16_t,16>) tmp).pack<TT>();
            int *tmpint = (int *) &(tmpout);
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConv1x1PktStream");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv1x1_0<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
  input_window<TT>* in,
  output_stream<acc48>* restrict cout
) {
  PROFILE_HEADER2;

  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  wzero = upd_v(wzero, 0, aie::broadcast<int16_t,8>(w_zero));
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_bias;

  v16w* restrict w_ptr = (v16w *) weights;
  TT* restrict in_ptr = (TT *) in->ptr;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      acc_bias.from_vector(aie::broadcast<int32_t, 16>(bias[m]), 0);
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          acc1 = acc_bias;
          
          for (int c = 0; c <= C_PER_M-16; c+=16) {
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            for (int i = 0; i < 16; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 16 != 0) {
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            for (int i = 0; i <= C_PER_M-2; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 2 != 0) {
            data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
            v16 zeros = aie::zeros<TT,16>();
            data = upd_v(data, 1, zeros);
            MAC_ROW(acc1, LAST_C);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel-C_PER_M, right 16
          w_ptr -= CKK_ROW_SIZE/16;

          writeincr_v8(cout, ext_lo(acc1));
          writeincr_v8(cout, ext_hi(acc1));
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += CKK_ROW_SIZE/16;
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConv1x1_0");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv1x1_1<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
  input_window<TT>* in,
  input_stream<acc48>* restrict cin,
  output_stream<acc48>* restrict cout
) {
  PROFILE_HEADER2;

  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  wzero = upd_v(wzero, 0, aie::broadcast<int16_t,8>(w_zero));
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  
  v16w* restrict w_ptr = (v16w *) weights;
  TT* restrict in_ptr = (TT *) in->ptr;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          aie::accum<acc48,8> _acc1 = readincr_v8(cin);
          aie::accum<acc48,8> _acc2 = readincr_v8(cin);
          acc1 = aie::concat(_acc1, _acc2);
          
          for (int c = 0; c <= C_PER_M-16; c+=16) {
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            for (int i = 0; i < 16; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 16 != 0) {
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            for (int i = 0; i <= C_PER_M-2; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 2 != 0) {
            data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
            v16 zeros = aie::zeros<TT,16>();
            data = upd_v(data, 1, zeros);
            MAC_ROW(acc1, LAST_C);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel-C_PER_M, right 16
          w_ptr -= CKK_ROW_SIZE/16;

          writeincr_v8(cout, ext_lo(acc1));
          writeincr_v8(cout, ext_hi(acc1));
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += CKK_ROW_SIZE/16;
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConv1x1_1");
}


template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
QLinearConv1x1_2<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::QLinearConv1x1_2(
  TTPARAM (&w)[M*CKK_ROW_SIZE],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TT y_zero
):
  weights(w), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), y_zero(y_zero)
{ 
  if ((std::is_same<TTPARAM,int8_t>::value)) assert(w_zero == 0);
  // -1 due to rounding, -1 to fit in 16b
  scalebits = 15 - log(x_scale*w_scale/y_scale) / log(2);
  assert(scalebits <= 30); // KH*KW*int8*int8*scale <= acc48, for KH=KW=1
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
}

template <typename TT, typename TTPARAM, int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W, int B, int C, int M, int KH, int KW, int GROUP>
void QLinearConv1x1_2<TT, TTPARAM, INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP>::filter(
  input_window<TT>* in,
  input_stream<acc48>* restrict cin,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;
  
  v32int16 wvec = null_v32int16();
  v32int16 wzero = null_v32int16();
  wzero = upd_v(wzero, 0, aie::broadcast<int16_t,8>(w_zero));
  v32 data = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w* restrict w_ptr = (v16w *) weights;
  TT* restrict in_ptr = (TT *) in->ptr;
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(M,) { 

      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W_PAD; w+=16/STEP_W) {

          aie::accum<acc48,8> _acc1 = readincr_v8(cin);
          aie::accum<acc48,8> _acc2 = readincr_v8(cin);
          acc1 = aie::concat(_acc1, _acc2);
          
          for (int c = 0; c <= C_PER_M-16; c+=16) {
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            for (int i = 0; i < 16; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 16 != 0) {
            wvec = upd_w(wvec, 0, unpack(*w_ptr)); w_ptr++;
            for (int i = 0; i <= C_PER_M-2; i+=2) {
              data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
              MAC_ROW(acc1, i);
            }
          }
          if (C_PER_M % 2 != 0) {
            data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
            v16 zeros = aie::zeros<TT,16>();
            data = upd_v(data, 1, zeros);
            MAC_ROW(acc1, LAST_C);
          }
          in_ptr += -C_PER_M*INP_H*INP_W + 16; // go channel-C_PER_M, right 16
          w_ptr -= CKK_ROW_SIZE/16;
          
          acc1 = aie::mac(acc_shift, (aie::vector<int32_t,16>) lsrs(acc1, 0), scale);
          if (STEP_W == 2) {
            v16int16 tmp = srs(acc1, scalebits);
            v8int16 tmphalf = aie::filter_even((aie::vector<int16_t,16>) tmp, 1);
            tmp = upd_v(tmp, 0, tmphalf);
            aie::vector<TT,16> tmpout = ((aie::vector<int16_t,16>) tmp).pack<TT>();
            int *tmpint = (int *) &(tmpout);
            put_ms(0, tmpint[0]);
            put_ms(0, tmpint[1]);
          } else {
            writeincr_v16(out, ((aie::accum<acc48,16>) acc1).to_vector<TT>(scalebits));
          }
        } // W
        
        in_ptr += INP_W*STEP_H - OUT_W_PAD*STEP_W; // go left OUT_W_PAD*STEP_W, down STEP_H
        chess_separator_scheduler(); // uncomment if compiler cannot detect out dependency
      } // H
      in_ptr -= INP_W*OUT_H*STEP_H; // go up OUT_H*STEP_H
      w_ptr += CKK_ROW_SIZE/16;
      if ((m % (M/GROUP)) == (M/GROUP - 1)) {
        in_ptr += C_PER_M*INP_H*INP_W;
      }
    } // M
  } // B

  CONV_PROFILE_FOOTER("QLinearConv1x1_2");
}
#undef MAC_ROW
