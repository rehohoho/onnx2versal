#include "qgemm.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


template <int M, int K, int N, int NPAD>
void QgemmScalar<M, K, N, NPAD>::filter(
	input_window<int8_t>* in,      // MxK
                                 // KxNPAD
  output_window<int8_t>* out     // MxNPAD
) {
  PROFILE_HEADER(printf(
    "Running QgemmScalar<%d,%d,%d,%d>\n", M, K, N, NPAD));

  int weightIdx = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int res = bias[j];
      weightIdx = j;
      for (int k = 0; k < K; k++) {
        int a = window_readincr(in);
        int b = weights[weightIdx];
        weightIdx += NPAD;
        res += (a-x_zero) * (b-w_zero);
      }
      res = y_zero + round(scale * res);
      res = std::min(std::max(res, -128), 127);
      window_writeincr(out, (int8_t) res);
      window_incr(in, -K); // repeat same in row for next j
    }
    for (int j = N; j < NPAD; j++)
      window_writeincr(out, y_zero);
    
    window_incr(in, K); // next in row for next N
  }

  PROFILE_FOOTER;
}


template <int M, int K, int N, int NPAD>
QgemmVector<M,K,N,NPAD>::QgemmVector (
  int8_t (&w)[K*NPAD],
  int32_t (&b)[NPAD],
  float x_scale,
  float w_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t y_zero
): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero(x_zero), w_zero(w_zero), y_zero(y_zero) {
  
  int8_t *w_ptr = (int8_t *) weights;
  int32_t *b_ptr = (int32_t *) bias;
  aie::accum<acc48,16> aieacc;

  // precompute x_zero_weights into bias
  for (int j = 0; j < N; j+=16) {
    aieacc.from_vector(aie::load_v<16>(b_ptr), 0);
    for (int k = 0; k < K; k++) {
      aieacc = aie::msc(aieacc, aie::load_v<16>(w_ptr), x_zero); w_ptr += NPAD;
    }
    aie::store_v(b_ptr, aieacc.to_vector<int32_t>(0)); 
    b_ptr += 16;
    w_ptr -= K*NPAD;
  }
  
  // -1 due to rounding, -1 to fit in 16b
  scalebits = std::abs(log(x_scale*w_scale/y_scale) / log(2)) + 15;
  assert(scalebits <= 24); // since int32_t shift, int8_t y_zero_point

  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
  shift = float2fix((float) y_zero, scalebits); // scalebits <= 24
};

/**
 * QgemmVector<28,32,24,32,1,1,6,5>
 * 
 * https://docs.xilinx.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-on-8x8-bits
 * int8 * int8 requires x indexing %4, z indexing %2
 * 
 *          z0  z1  z2  z3  z4  z5  z6  z7
 * acc0  += x0  x16 x32 x48 x64 x80 x96 x112
 * acc1  += x1  x17
 * acc2  += x2  x18
 * acc3  += x3  x19
 * acc4  += x4  x20
 * acc5  += x5  x21
 * acc6  += x6  x22
 * acc7  += x7  x23
 * acc8  += x8  x24
 * acc9  += x9  x25
 * acc10 += x10 x26
 * acc11 += x11 x27
 * acc12 += x12 x28
 * acc13 += x13 x29
 * acc14 += x14 x30
 * acc15 += x15 x31
 * 
 * xidx: 30 => 0-3, 16-19, 33 => 12-15, 28-31 (square executes on 4x2 matrix)
 * zidx: 00 => 0,1, 2,3 => 0 1 0 1
 * acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
 * 
 * Vector registers can hold 256 int8 at most, 128 int16 at most.
 */
template <int M, int K, int N, int NPAD>
void QgemmVector<M, K, N, NPAD>::filter(
	input_window<int8_t>* in,      // MxK
                                 // KxNPAD
  output_window<int8_t>* out     // MxNPAD
) {
  PROFILE_HEADER(printf(
    "Running QgemmVector<%d,%d,%d,%d>\n", M, K, N, NPAD));

  int8_t *in_ptr = (int8_t *) in->ptr;
  int8_t *w_ptr = (int8_t *) weights;
  int32_t *b_ptr = (int32_t *) bias;
  v16int8 *out_ptr = (v16int8 *) out->ptr;

  v128int8 wmat = null_v128int8();
  v32int8 inmat = null_v32int8();
  v16int32 accbuf1 = undef_v16int32();
  aie::accum<acc48,16> aieacc1;

  v16acc48 acc1 = undef_v16acc48();

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

#define LOAD_W \
  wmat = upd_v(wmat, 0, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 1, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 2, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 3, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 4, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 5, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 6, *(v16int8 *) w_ptr); w_ptr += NPAD; \
  wmat = upd_v(wmat, 7, *(v16int8 *) w_ptr); w_ptr += NPAD;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j+=16) {
      
      acc1 = null_v16acc48();
      int k = 0;

      for (k; k <= K-16; k+=16) { // += input[k:k+16] * weight[k:k+8,n:n+16]
        inmat = upd_v(inmat, 0, *(v16int8 *) in_ptr); in_ptr += 16; // load input[k:k+8]
        LOAD_W; // load weight[k:k+8,n:n+16]
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
        LOAD_W; // load weight[k+8:k+16,n:n+16]
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 8, 0x00000000, 2, 0x1010);
      } // K-16
      for (k; k <= K-8; k+=8) {
        inmat = upd_v(inmat, 0, *(v16int8 *) in_ptr); in_ptr += 8;
        LOAD_W;
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
      } // K-8
      for (k; k <= K-4; k+=4) {
        inmat = upd_v(inmat, 0, *(v16int8 *) in_ptr); in_ptr += 4;
        wmat = null_v128int8();
        wmat = upd_v(wmat, 0, *(v16int8 *) w_ptr); w_ptr += NPAD;
        wmat = upd_v(wmat, 1, *(v16int8 *) w_ptr); w_ptr += NPAD;
        wmat = upd_v(wmat, 2, *(v16int8 *) w_ptr); w_ptr += NPAD;
        wmat = upd_v(wmat, 3, *(v16int8 *) w_ptr); w_ptr += NPAD;
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
      } // K-4
      for (k; k < K; k+=2) {
        inmat = upd_v(inmat, 0, *(v16int8 *) in_ptr); in_ptr += 2;
        wmat = null_v128int8();
        wmat = upd_v(wmat, 0, *(v16int8 *) w_ptr); w_ptr += NPAD;
        wmat = upd_v(wmat, 1, *(v16int8 *) w_ptr); w_ptr += NPAD;
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
      } // K

      acc1 = aie::add((aie::accum<acc48,16>) acc1, aie::load_v<16>(b_ptr)); b_ptr += 16;
      accbuf1 = lsrs(acc1, 0);
      aieacc1 = aie::mul<acc48>((aie::vector<int32_t,16>) accbuf1, scale);
      aieacc1 = aie::add(aieacc1, shift);
      *out_ptr = aieacc1.to_vector<int8_t>(scalebits);
      out_ptr++;

      in_ptr -= K;          // reset
      w_ptr -= K*NPAD + 16; // next
    } // N
    in_ptr += K;      // next
    b_ptr -= N/16*16; // reset
    w_ptr -= NPAD;    // reset
  } // M

  PROFILE_FOOTER;
}
