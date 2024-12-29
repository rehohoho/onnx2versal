#include "qlinearconv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define CONV_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), typeid(TTPARAM).name(), INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, B, C, M, KH, KW, GROUP);

/**
 * acc0  += z0  z16
 * acc1  += z1  z17
 * ...
 * acc14 += z14
 * acc15 += z15
 * 
 * xoffsets: 4b offset for every two lanes, e.g. offset=30 => 0*4=0, (3+0+1)*4=16 => 0,2, 16,18
 *                                                                                   1,3, 17,19
 * 
 * acc0  += 0
 * acc1  += 10
 * acc2  += 
 * acc3  += 
 * acc4  += 2
 * acc5  += 12
 * acc6  += 
 * acc7  += 
 * acc8  += 4
 * acc9  += 
 * acc10 += 
 * acc11 += 
 * acc12 += 8
 * acc13 += 
 * acc14 += 
 * acc15 += 
 * 
 * 0  0   0
 * 8  4   2
 * 1  8   4
 * 9  12  6
 * 2  1
 * 10 5
 * 3  9
 * 11 13
 * 4  2
 * 12 6
 * 5  10
 * 13 14
 * 6  3
 * 14 7
 * 7  11
 * 15 15
 */
#define MAC_ROW(acc, widx) \
  acc = mac16(acc, data, 0, 0x33323130, 32, 0x3120, wvec, widx, 0, 2, 0x1010);

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

  using v64 = typename std::conditional<(std::is_same<TT, int8_t>::value), v64int8, v64uint8>::type;
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  using v16w = typename std::conditional<(std::is_same<TTPARAM, int8_t>::value), v16int8, v16uint8>::type;

  v64 data = aie::zeros<TT,64>();
  v32 wvec = aie::zeros<TT,32>();

  v16acc48 acc1 = undef_v16acc48();
  aie::accum<acc48,16> acc_shift;
  aie::accum<acc48,16> acc_bias;
  acc_shift.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16w* restrict ckk_row_ptr = (v16w *) ckk_row;
  TT* restrict in_ptr = (TT *) in->ptr;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

#define LOAD_W \
  data = upd_v(data, 0, *(v16 *) in_ptr); in_ptr += INP_H*INP_W; \
  data = upd_v(data, 1, *(v16 *) in_ptr); in_ptr += INP_H*INP_W; \
  data = upd_v(data, 2, *(v16 *) in_ptr); in_ptr += INP_H*INP_W; \
  data = upd_v(data, 3, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;

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
          
          for (int c = 0; c < CKK_ROW_SIZE - 16; c+=16) {
            wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
            LOAD_W; MAC_ROW(acc1, 0);
            LOAD_W; MAC_ROW(acc1, 4);
            LOAD_W; MAC_ROW(acc1, 8);
            LOAD_W; MAC_ROW(acc1, 12);
          }
          // remaining C in [0, 15]
          wvec = upd_v(wvec, 0, *ckk_row_ptr); ckk_row_ptr++;
          if (C % 16 >= 4 || C % 16 == 0) { LOAD_W; MAC_ROW(acc1, 0); }
          if (C % 16 >= 8 || C % 16 == 0) { LOAD_W; MAC_ROW(acc1, 4); }
          if (C % 16 >= 12 || C % 16 == 0) { LOAD_W; MAC_ROW(acc1, 8); }
          if (C % 4 != 0) {
            for (int i = 0; i < C % 4; i++) {
              data = upd_v(data, i, *(v16 *) in_ptr); in_ptr += INP_H*INP_W;
            }
            MAC_ROW(acc1, (C % 16)/4*4);
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
#undef MAC_ROW