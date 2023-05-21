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
        for (int w = 0; w < OUT_W; w++) {
        
          // qy = qy_zero + [(qx-qx_zero)*(qw-qw_zero) + qbias] * qx_scale*qw_scale/qy_scale
          int res = bias[m];
          weightIdx = m*C*K*16;
          
          for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
              for (int q = 0; q < K; q++) {
                int a = window_readincr(in);
                res += (a - x_zero_point) * (weights[weightIdx+q]-w_zero_point);
              }
              weightIdx += 16;
              window_incr(in, -K+INP_W); // go left K, down 1
            }
            window_incr(in, -K*INP_W + INP_H*INP_W); // go up K, channel 1
          }
          res = y_zero_point + round(x_scale*w_scale/y_scale * res);
          
          // saturate at the end only
          res = std::min(std::max(res, -128), 128);

          // if (res < 0) res = 0;
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
  int8_t *w_ptr = (int8_t *) weights;
  
  // precompute x_zero_weights into bias
  for (int m = 0; m < M; m++) {
    int res = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        v16int16 wvec = unpack(*(v16int8 *) w_ptr); w_ptr += 16;
        for (int q = 0; q < K; q++) {
          // printf("%d:%d ", x_zero_point, ext_elem(wvec, q));
          res += x_zero_point * ext_elem(wvec, q);
        }
      }
    }
    // printf("%d\n", res);
    bias[m] -= res;
  }
}

/**
 * unpack to int16 to handle   OOB int8-int8
 *
 * mac16 (v16acc48 acc, 
 *  v32int16 xbuff, int xstart, unsigned int xoffsets, unsigned int xoffsets_hi, unsigned int xsquare, 
 *  v16int16 zbuff, int zstart, unsigned int zoffsets, unsigned int zoffsets_hi, int zstep)
 * 
 * https://docs.xilinx.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-on-8x8-bits
 * 24 selects 4*4=16, (4+2+1)*4=28 => rows (16,18),(17,19),(28,30),(29,30) before square
 * square executes on 4x2 matrix
 * 
 * 
 * mac16 (v16acc48 acc, 
 *  v64int16 xbuff, int xstart, unsigned int xoffsets, unsigned int xoffsets_hi, int xstep, unsigned int xsquare, 
 *  v32int8 zbuff, int zstart, unsigned int zoffsets, unsigned int zoffsets_hi, int zstep, unsigned int zsquare)
 * Requires: x indexing
 * 
 * acc0  += z0*x0  + z1*x1   z2*x2  + z3*x3
 * acc1  += z0*x1  + z1*x2   z2*x3  + z3*x4
 * acc2  += z0*x2  + z1*x3   z2*x4  + z3*x5
 * acc3  += z0*x3  + z1*x4   z2*x5  + z3*x6
 * acc4  += z0*x4  + z1*x5   z2*x6  + z3*x7
 * acc5  += z0*x5  + z1*x6   z2*x7  + z3*x8
 * acc6  += z0*x6  + z1*x7   z2*x8  + z3*x9
 * acc7  += z0*x7  + z1*x8   z2*x9  + z3*x10
 * acc8  += z0*x8  + z1*x9   z2*x10 + z3*x11
 * acc9  += z0*x9  + z1*x10  z2*x11 + z3*x12
 * acc10 += z0*x10 + z1*x11  z2*x12 + z3*x13
 * acc11 += z0*x11 + z1*x12  z2*x13 + z3*x14
 * acc12 += z0*x12 + z1*x13  z2*x14 + z3*x15
 * acc13 += z0*x13 + z1*x14  z2*x15 + z3*x16
 * acc14 += z0*x14 + z1*x15  z2*x16 + z3*x17
 * acc15 += z0*x15 + z1*x16  z2*x17 + z3*x18
 * 
 * 
 * mac16 (v16acc48 acc, 
 *  v64int8 xbuff, int xstart, unsigned int xoffsets, int xstep, unsigned int xsquare, 
 *  v32int8 zbuff, int zstart, unsigned int zoffsets, int zstep, unsigned int zsquare)
 * Requires: x indexing %4, z indexing %2
 * v64int8, v32int8 can store 2 rows of calculations, 2 mac16 / 4 loads
 * 
 * acc0  += z0*x0  + z1*x1   z2*x2  + z3*x3   z4*x4
 * acc1  += z0*x1  + z1*x2   z2*x3  + z3*x4   z4*x5
 * acc2  += z0*x2  + z1*x3   z2*x4  + z3*x5   z4*x6  x2 %4 != 0, not indexable
 * acc3  += z0*x3  + z1*x4   z2*x5  + z3*x6   z4*x7
 * acc4  += z0*x4  + z1*x5   z2*x6  + z3*x7   z4*x8
 * acc5  += z0*x5  + z1*x6   z2*x7  + z3*x8   z4*x9
 * acc6  += z0*x6  + z1*x7   z2*x8  + z3*x9   z4*x10
 * acc7  += z0*x7  + z1*x8   z2*x9  + z3*x10  z4*x11
 * acc8  += z0*x8  + z1*x9   z2*x10 + z3*x11  z4*x12
 * acc9  += z0*x9  + z1*x10  z2*x11 + z3*x12  z4*x13
 * acc10 += z0*x10 + z1*x11  z2*x12 + z3*x13  z4*x14
 * acc11 += z0*x11 + z1*x12  z2*x13 + z3*x14  z4*x15
 * acc12 += z0*x12 + z1*x13  z2*x14 + z3*x15  z4*x16
 * acc13 += z0*x13 + z1*x14  z2*x15 + z3*x16  z4*x17
 * acc14 += z0*x14 + z1*x15  z2*x16 + z3*x17  z4*x18
 * acc15 += z0*x15 + z1*x16  z2*x17 + z3*x18  z4*x19
 * 
 * int8xint8, requires to expand 5 weights into 16 long vector (idx 4-12)
 * acc0  += x4*z0  + x5*z1   x8*z2  + x9*z3   x12*z4 
 * acc1  += x4*z1  + x5*z2   x8*z3  + x9*z4   x12*z5
 * acc2  +=                  x4*z2  + x5*z3   x8*z4  + x9*z5   x12*z6
 * acc3  +=                  x4*z3  + x5*z4   x8*z5  + x9*z6   x12*z7
 * 
 * acc4  += x4*z4  + x5*z5   x8*z6  + x9*z7   x12*z8
 * acc5  += x4*z5  + x5*z6   x8*z7  + x9*z8   x12*z9
 * acc6  +=                  x4*z6  + x5*z7   x8*z8  + x9*z9   x12*z10
 * acc7  +=                  x4*z7  + x5*z8   x8*z9  + x9*z10  x12*z11
 * 
 * acc8  += x4*z8  + x5*z9   x8*z10 + x9*z11  x12*z12
 * acc9  += x4*z9  + x5*z10  x8*z11 + x9*z12  x12*z13
 * acc10 +=                  x4*z10 + x5*z11  x8*z12 + x9*z13  x12*z14
 * acc11 +=                  x4*z11 + x5*z12  x8*z13 + x9*z14  x12*z15
 * 
 * acc12 += x4*z12 + x5*z13  x8*z14 + x9*z15  x12*z16
 * acc13 += x4*z13 + x5*z14  x8*z15 + x9*z16  x12*z17
 * acc14 +=                  x4*z14 + x5*z15  x8*z16 + x9*z17  x12*z18
 * acc15 +=                  x4*z15 + x5*z16  x8*z17 + x9*z18  x12*z19
 * 
 * v64int8 wvec; // x, indexable in *4
 * v32int8 data; // z, indexable in *2
 * acc = mac16(acc, wvec, 0, xoffsets, xstep, xsquare, data, 0, 0x06040200, 2, 0x2110);
 * 
 * 
 * Vector registers can hold 256 int8 at most, 128 int16 at most.
 * Use pointers to avoid alignment issues on loads
 * 
 * Requires INP_W%16=0, OUT_W%16=0
 */
template <int INP_H, int INP_W, int OUT_H, int OUT_W, int B, int C, int M, int K>
void QLinearConvVector<INP_H, INP_W, OUT_H, OUT_W, B, C, M, K>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QLinearConvVector<%d,%d,%d,%d,%d,%d,%d,%d>\n", INP_H, INP_W, OUT_H, OUT_W, B, C, M, K));
  
  v16int16 add = aie::broadcast<int16_t, 16>(y_zero_point);
  v8float scale = aie::broadcast<float, 8>(x_scale*w_scale/y_scale);
  v32int16 data = undef_v32int16();
  v16int16 wvec = undef_v16int16();
  v16int16 res = null_v16int16();
  v8acc48 resacc = null_v8acc48();

  // print_vec<short, short>((short *) &sub, 16);
  // print_fvec<float>((float *) &scale, 8);
  int8_t *in_ptr = (int8_t *) in->ptr;
  int8_t *w_ptr = (int8_t *) weights;
  const int W_REM = OUT_H % 16;
  const int SELECT_S = ((1 << W_REM) - 1) << (16 - W_REM);

// xoffsets: 4b offset for lane 0,2,4,6, for 04, off0=2*4, off2=(0+4 +1)*2 => 8,9, 10,11
#define MAC_ROW \
  acc1 = mac16(acc1, data, 0, 0x03020100, 0x07060504, 0x2110, wvec, 0, 0x0, 0x0, 1); \
  acc1 = mac16(acc1, data, 0, 0x04030201, 0x08070605, 0x2110, wvec, 2, 0x0, 0x0, 1); \
  acc1 = mac16(acc1, data, 0, 0x05040302, 0x09080706, 0x2110, wvec, 4, 0x0, 0x0, 1); // expect 6th to be 0
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { 
      v16int32 bvec = aie::broadcast<int32_t, 16>(bias[m]);
      for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w+=16) {

          // qy = qy_zero + [(qx-qx_zero)*(qw-qw_zero) + qbias] * qx_scale*qw_scale/qy_scale
          v16acc48 acc1 = null_v16acc48(); // ups(bvec, 0);
        
          for (int c = 0; c < C; c++) { // computes 1x16 partial products over 5x5 kernel
            data = unpack(*(v32int8 *)in_ptr); in_ptr += INP_W;
            wvec = unpack(*(v16int8 *)w_ptr); w_ptr += 16;
            MAC_ROW;
            
            data = unpack(*(v32int8 *)in_ptr); in_ptr += INP_W;
            wvec = unpack(*(v16int8 *)w_ptr); w_ptr += 16;
            MAC_ROW;

            data = unpack(*(v32int8 *)in_ptr); in_ptr += INP_W;
            wvec = unpack(*(v16int8 *)w_ptr); w_ptr += 16;  
            MAC_ROW;
            
            data = unpack(*(v32int8 *)in_ptr); in_ptr += INP_W;
            wvec = unpack(*(v16int8 *)w_ptr); w_ptr += 16;
            MAC_ROW;
            
            data = unpack(*(v32int8 *) in_ptr); in_ptr += INP_H*INP_W - 4*INP_W; // channel +1, up 1
            wvec = unpack(*(v16int8 *)w_ptr); w_ptr += 16;
            MAC_ROW;
          }

          v16int32 accbuf = lsrs(acc1, 0) + bvec; // cast to int32
          if (w + 16 >= OUT_W) {
            accbuf = select16(SELECT_S, accbuf, null_v16int32());
          }
          
          // fix2float, fpmac, float2fix takes stupidly long, use fixed point mult
          v8float facc = fix2float(ext_w(accbuf, 0));
          facc = fpmul(facc, scale);
          v8int32 halfacc = float2fix(facc, 15);
          resacc = ups(halfacc, 0);
          res = upd_v(res, 0, srs(resacc, 15));

          facc = fix2float(ext_w(accbuf, 1));
          facc = fpmul(facc, scale);
          halfacc = float2fix(facc, 15);  
          resacc = ups(halfacc, 0);
          res = upd_v(res, 1, srs(resacc, 15));
          res = res + add;

          in_ptr += 16 - C*INP_H*INP_W; // go channel-C, right 16
          
          // convert v16int32 to v16int16 =.=
          // v32int16 accbuf2 = *(v32int16 *) &accbuf;
          // print_vec<short, short>((short *) &accbuf2, 32);
          // accbuf2 = shuffle32(accbuf2, 0, 0x06040200, 0x0e0c0a08, 0x3120); // 0213 4657 ....
          // accbuf2 = shuffle32(accbuf2, 0, 0x1c181410, 0x00000000, 0x3210); // 0246 8ace ....
          
          // v16int16 accbuf3 = ext_w(accbuf2, 0) + add;
          // print_vec<short, short>((short *) &accbuf3, 16);

          // acc1 = ups(accbuf3, 0);
          // v16int16 acc1buf = srs(acc1, 8);
          // print_vec<short, short>((short *) &acc1buf, 16);
          // window_writeincr(out, bsrs(acc1 , 8)); // v16int8
          window_writeincr(out, pack(res)); // v16int8
          w_ptr -= C*16*5;
        } // W

        in_ptr += INP_W - OUT_W; // go left OUT_W, down 1
        // window_incr(out, OUT_W);
        // printf("\n");
      } // H
      in_ptr -= OUT_H*INP_W; // go up OUT_H
      w_ptr += C*16*5;
    } // M
  } // B


  PROFILE_FOOTER;
}
