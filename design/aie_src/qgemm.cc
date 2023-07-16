#include "qgemm.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define QGEMM_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d>", \
    filter_name, typeid(TT).name(), M, K, N);

template <typename TT, int M, int K, int N>
void QgemmScalarStream<TT, M, K, N>::filter(
	input_window<TT>* in,      // MxK
                                 // KxN
  output_stream<TT>* out     // MxN
) {
  PROFILE_HEADER2;

  int weightIdx = 0;
  int resvi = 0;
  v16int16 resv = null_v16int16();

#define WRITE_OUT(res) \
  resv = upd_elem(resv, resvi, res); \
  if (resvi == 15) writeincr_v16(out, ((aie::vector<int16,16>) resv).pack<TT>()); \
  resvi = (resvi + 1) & 0xf;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int res = bias[j];
      weightIdx = j;
      for (int k = 0; k < K; k++) {
        int a = window_readincr(in);
        int b = weights[weightIdx];
        weightIdx += N;
        res += a * (b-w_zero);
      }
      res = y_zero + round(scale * res);
      if ((std::is_same<TT, int8_t>::value)) {
        res = std::min(std::max(res, -128), 127);
      } else {
        res = std::min(std::max(res, 0), 255);
      }
      WRITE_OUT(res);
      window_incr(in, -K); // repeat same in row for next j
    }
    window_incr(in, K); // next in row for next N
  }
#undef WRITE_OUT

  QGEMM_PROFILE_FOOTER("QgemmScalarStream");
}


template <typename TT, int M, int K, int N>
QgemmStream<TT, M, K, N>::QgemmStream (
  int8_t (&w)[K*N],
  int32_t (&b)[N],
  float x_scale,
  float w_scale,
  float y_scale,
  TT x_zero,
  int8_t w_zero,
  TT y_zero
): weights(w), bias(b), x_scale(x_scale), w_scale(w_scale), y_scale(y_scale), x_zero(x_zero), w_zero(w_zero), y_zero(y_zero) {
  // -1 due to rounding, -1 to fit in 16b
  scalebits = std::abs(log(x_scale*w_scale/y_scale) / log(2)) + 15;
  if (scalebits >= 48 - log(K)/log(2) - 16)
    printf("WARNING: scalebits %d vs %f \n", scalebits, 48 - log(K)/log(2) - 16); // K*int8*int8*scale <= acc48
  scale = float2fix(x_scale*w_scale/y_scale, scalebits);
};

/**
 * QgemmStream<28,32,24,32,1,1,6,5>
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
template <typename TT, int M, int K, int N>
void QgemmStream<TT, M, K, N>::filter(
	input_window<TT>* in,      // MxK
                                 // KxN
  output_stream<TT>* out     // MxN
) {
  PROFILE_HEADER2;

  using v128 = typename std::conditional<(std::is_same<TT, int8_t>::value), v128int8, v128uint8>::type;
  using v32 = typename std::conditional<(std::is_same<TT, int8_t>::value), v32int8, v32uint8>::type;
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;

  TT *in_ptr = (TT *) in->ptr;
  int8_t *w_ptr = (int8_t *) weights;
  int32_t *b_ptr = (int32_t *) bias;

  v128 wmat = aie::zeros<TT,128>();
  v32 inmat = aie::zeros<TT,32>();
  aie::accum<acc48,16> aieacc1;
  aie::accum<acc48,16> acc_shift1;
  acc_shift1.from_vector(aie::broadcast<int16_t, 16>(y_zero), scalebits);

  v16acc48 acc1 = undef_v16acc48();

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

#define LOAD_W \
  wmat = upd_v(wmat, 0, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 1, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 2, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 3, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 4, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 5, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 6, *(v16 *) w_ptr); w_ptr += N; \
  wmat = upd_v(wmat, 7, *(v16 *) w_ptr); w_ptr += N;

  for (int i = 0; i < M; i++) chess_prepare_for_pipelining chess_loop_range(M,) {
    for (int j = 0; j < N; j+=16) chess_prepare_for_pipelining chess_loop_range(N/16,) {
      
      acc1 = null_v16acc48();

      for (int k = 0; k <= K-16; k+=16) { // += input[k:k+16] * weight[k:k+8,n:n+16]
        inmat = upd_v(inmat, 0, *(v16 *) in_ptr); in_ptr += 16; // load input[k:k+8]
        LOAD_W; // load weight[k:k+8,n:n+16]
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
        LOAD_W; // load weight[k+8:k+16,n:n+16]
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 8, 0x00000000, 2, 0x1010);
      } // K-16
      if (RUN_8CHUNK) {
        inmat = upd_v(inmat, 0, *(v16 *) in_ptr); in_ptr += 8;
        LOAD_W;
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
      } // K-8
      if (RUN_LASTCHUNK) {
        inmat = upd_v(inmat, 0, *(v16 *) in_ptr); in_ptr += K_REM16;
        wmat = aie::zeros<TT,128>();
        for (int p = 0; p < K_REM16; p++) {
          wmat = upd_v(wmat, p, *(v16 *) w_ptr); w_ptr += N;
        }
        acc1 = mac16(acc1, wmat, 0, 0x33323130, 32, 0x3120, inmat, 0, 0x00000000, 2, 0x1010);
      } // K

      aieacc1 = aie::add((aie::accum<acc48,16>) acc1, aie::load_v<16>(b_ptr)); b_ptr += 16;
      aieacc1 = aie::mac(acc_shift1, aieacc1.to_vector<int32_t>(0), scale);
      writeincr_v16(out, aieacc1.to_vector<TT>(scalebits));

      in_ptr -= K;        // reset
      w_ptr += -K*N + 16; // next
    } // N
    
    // Error when M > 1
    // Internal error: chess-backend: mist/block_depcies.cpp:1481: int min_distance_to(const CFG&, const Bundle*, Macro_node*): Assertion `n' failed.
    chess_separator_scheduler();

    in_ptr += K;
    b_ptr -= N; // reset
    w_ptr -= N;    // reset
  } // M

#undef LOAD_W

  QGEMM_PROFILE_FOOTER("QgemmStream");
}
