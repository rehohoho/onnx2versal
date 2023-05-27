#include "qlinearconv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
void QLinearConvScalar<INP_H, INP_W, OUT_H, OUT_W, B, C, M, K>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QLinearConvScalar<%d,%d,%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C, M, K));

  int weightIdx = 0;

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_H; w++) {
        
          int res = bias[m];
          weightIdx = m*C*K*16;
          
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              // for (int q = 0; q < K; q++) {
              for (int q = 4; q < 4 + K*2; q+=2) {
                int a = window_readincr(in);
                res += (a - x_zero_point) * (weights[weightIdx+q]-w_zero_point);
              }
              weightIdx += 16;
              window_incr(in, -K+INP_W); // go left K, down 1
            }
            window_incr(in, -K*INP_W + INP_H*INP_W); // go up K, channel 1
          }
          res = y_zero_point + round(scale * res);
          res = std::min(std::max(res, -128), 127);

          window_writeincr(out, (int8_t) res);
          window_incr(in, -C*INP_H*INP_W + 1); // go channel -C, right 1
        }
        window_incr(out, OUT_W-OUT_H);
        window_incr(in, INP_W-OUT_H); // go left OUT_H, go down 1
      }
      window_incr(in, -OUT_H*INP_W); // go up OUT_H
    }
  }

  PROFILE_FOOTER;
}



template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
QLinearConvVector<INP_H, INP_W, OUT_H, OUT_W, B, C, M, K>::QLinearConvVector(
  int8_t (&w)[M*C*K*16],
  int32_t (&b)[M],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero_point,
  int8_t w_zero_point,
  int8_t y_zero_point
):
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), 
  x_zero_point(x_zero_point), w_zero_point(w_zero_point), y_zero_point(y_zero_point)
{ 
  // qy = qy_zero + [(qx-qx_zero)*(qw-qw_zero) + qbias] * qx_scale*qw_scale/qy_scale
  v16int8 *w_ptr = (v16int8 *) weights;
  
  // precompute x_zero_weights into bias
  for (int m = 0; m < M; m++) {
    int res = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        v16int16 wvec = unpack(*w_ptr); w_ptr++;
        // for (int q = 0; q < K; q++) {
        for (int q = 4; q < 4 + K*2; q+=2) {
          res += x_zero_point * ext_elem(wvec, q);
        }
      }
    }
    bias[m] -= res;
  }
  
  // -1 due to rounding, -1 to fit in 16b
  scalebits = std::abs(log(x_scale*w_scale/y_scale) / log(2)) + 15;
  scalebits = std::min(scalebits, 24); // since int32_t shift, int8_t y_zero_point

  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
  shift = float2fix((float) y_zero_point, scalebits); // scalebits <= 24,
}

/**
 * QLinearConvVector<28,32,24,32,1,1,6,5>
 * fp scale: ~40k
 * fixed point scale: ~14k
 * precompute x_zero_weight into bias: 11341
 * int16*int8: 9177
 * int8*int8: 6724
 * precompute y_zero: 5174
 * two accs: ~4541
 * remove redundant add, scale vectors: 4225
 * remove precomputing y_zero: ~4700
 * separate buffers for scaleshift: 4398
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
 * acc5  += x5*z5  + x7*z6   x9*z7  + x11*z8  x13*z9 (z6 reads - 128?) -128 94 -> -128 -128
 * acc6  +=                  x4*z6  + x6*z7   x8*z8  + x10*z9   x12*z10
 * acc7  +=                  x5*z7  + x7*z8   x9*z9  + x11*z10  x13*z11 (z8 reads -128?) 126 126 -> 126 -128
 * 
 * acc8  += x4*z8  + x6*z9   x8*z10 + x10*z11 x12*z12
 * acc9  += x5*z9  + x7*z10  x9*z11 + x11*z12 x13*z13 x
 * acc10 +=                  x4*z10 + x6*z11  x8*z12 + x10*z13  x12*z14
 * acc11 +=                  x5*z11 + x7*z12  x9*z13 + x11*z14  x13*z15 x
 * 
 * acc12 += x4*z12 + x6*z13  x8*z14 + x10*z15 x12*z16
 * acc13 += x5*z13 + x7*z14  x9*z15 + x11*z16 x13*z17 x
 * acc14 +=                  x4*z14 + x6*z15  x8*z16 + x10*z17  x12*z18
 * acc15 +=                  x5*z15 + x7*z16  x9*z17 + x11*z18  x13*z19 x
 * 
 * Vector registers can hold 256 int8 at most, 128 int16 at most.
 */
template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
void QLinearConvVector<INP_H, INP_W, OUT_H, OUT_W, B, C, M, K>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QLinearConvVector<%d,%d,%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C, M, K));
  
  v64int8 wvec = null_v64int8();
  v32int8 data = null_v32int8();
  v16int32 accbuf1 = undef_v16int32();
  v16int32 accbuf2 = undef_v16int32();

  v16acc48 acc1 = undef_v16acc48();
  v16acc48 acc2 = undef_v16acc48();
  aie::accum<acc48,16> aieacc1;
  aie::accum<acc48,16> aieacc2;

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
      for (int h = 0; h < OUT_H; h+=2) {
        for (int w = 0; w < OUT_W; w+=16) {

          acc1 = null_v16acc48();
          acc2 = null_v16acc48();
        
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
          acc1 = aie::add((aie::accum<acc48,16>) acc1, bias[m]);
          acc2 = aie::add((aie::accum<acc48,16>) acc2, bias[m]);
          accbuf1 = lsrs(acc1, 0);
          accbuf2 = lsrs(acc2, 0);
          aieacc1 = aie::mul<acc48>((aie::vector<int32_t,16>) accbuf1, scale);
          aieacc2 = aie::mul<acc48>((aie::vector<int32_t,16>) accbuf2, scale);
          aieacc1 = aie::add(aieacc1, shift);
          aieacc2 = aie::add(aieacc2, shift);
          *out_ptr = aieacc1.to_vector<int8_t>(scalebits);
          out_ptr += OUT_W/16;
          *out_ptr = aieacc2.to_vector<int8_t>(scalebits);
          out_ptr += 1-OUT_W/16;

          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
          w_ptr -= C*5;
        } // W
        
        in_ptr += 2*INP_W - OUT_W; // go left OUT_W, down 2
        out_ptr += OUT_W/16;
      } // H
      in_ptr -= OUT_H*INP_W; // go up OUT_H
      w_ptr += C*5;
    } // M
  } // B

#undef MAC_ROW

  PROFILE_FOOTER;
}
