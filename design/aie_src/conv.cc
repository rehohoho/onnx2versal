#include "conv.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


/*
weight+bias const: 249103 cycles (conv01), 392981 (conv03)
weight const: 314790 cycles
weight load: 93331*28=2613268 cycles
*/
template <int INP_W, int OUT_W, int B, int C, int M, int K>
void ConvReluScalarBHWC<INP_W, OUT_W, B, C, M, K>::filter(
	input_window<float>* in,      // BHWC (1x28x28x1)
  output_window<float>* out     // BHWM (1x24x24x6)
) {
  PROFILE_HEADER;
  printf("Running conv_relu_scalar<%d, %d, %d, %d, %d, %d>", INP_W, OUT_W, B, C, M, K);

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
  PROFILE_HEADER;
  printf("Running conv_relu_scalar<%d, %d, %d, %d, %d, %d>", INP_W, OUT_W, B, C, M, K);

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
                // if (w == 0) printf("%f ", a);
              }
              window_incr(in, -K+INP_W); // go left K, down 1
              // if (w == 0) printf("\n");
            }
            window_incr(in, -K*INP_W + INP_W*INP_W); // go up K, channel 1
          }
          // if (w == 0) printf("\n");

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


/*
Using:
v8float fpmac (v8float        acc,
		           v32float       xbuf,
               int  	        xstart,
               unsigned int  	xoffs,
               v8float  	    zbuf, 
               int  	        zstart, !! compile time constant if zbuf !!
               unsigned int  	zoffs)

for (i = 0; i < 8; i++)
  ret[i] = acc[i] + xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]]

8 outputs per loop:
in[(k*row)+i:(k*row)+i+8] * weights[k*K+i], 0<=i<=K

Reference performance: Lenet Tutorial (int8) example with matmul ~2k cycles
- Note zstart must be a compile time constant
- Using conditionals ~2x loop time, so shuffle down to handle %4 vs %5
- Loop order BMHW seems faster since H and W > M

lenet conv2 (1x16x12x12):
manual unroll: 46757 cycles
use upd_w instead of loading direct: 75421 cycles

direct load: 46449
upd_w: 40711 -> 39700
*/
#ifdef __X86SIM__
#define GET_WVEC(wp, zstart) \
  wvec = *(v8float*) wp;
#else
#define GET_WVEC(wp, zstart) \
  wvecc = aie::load_v<8>(wp); \
  wvec = aie::shuffle_down(wvecc, zstart);
#endif

template <int INP_W, int OUT_W, int B, int C, int M>
void Conv5x5ReluBCHW<INP_W, OUT_W, B, C, M>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER;
  printf("Running Conv5x5ReluBCHW<%d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M);

  v8float zeros = null_v8float();
  v8float wvec;
  v16float data = null_v16float();
  aie::vector<float, 8> wvecc;

  // BHWM
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(1,) {
    for (int m = 0; m < M; m++) chess_prepare_for_pipelining chess_loop_range(1,)  { // computes one output channel
      for (int h = 0; h < OUT_W; h++) chess_prepare_for_pipelining chess_loop_range(1,)  {
        for (int w = 0; w < OUT_W; w+=8) chess_prepare_for_pipelining chess_loop_range(1,)  { // computes 8 output channel pixels
          
          v8float acc = aie::broadcast<float, 8>(bias[m]);
          float* wp = weights + m*C*5*5;
          int zstart = m*C*5*5 & 0x3;

          // flatten to avoid pipelining this, TODO: compute multiple channels to allow pipelining
          for (int c = 0; c < C; c++) chess_flatten_loop { // computes 8 partial products over 5x5 kernel
            
            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            GET_WVEC(wp, zstart);
            wp += 5;
            zstart = (zstart + 1) & 0x3;
            acc = fpmac(acc, data, 0, 0x76543210, wvec, 0, 0x00000000);
            acc = fpmac(acc, data, 1, 0x76543210, wvec, 1, 0x00000000);
            acc = fpmac(acc, data, 2, 0x76543210, wvec, 2, 0x00000000);
            acc = fpmac(acc, data, 3, 0x76543210, wvec, 3, 0x00000000);
            acc = fpmac(acc, data, 4, 0x76543210, wvec, 4, 0x00000000);

            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            GET_WVEC(wp, zstart);
            wp += 5;
            zstart = (zstart + 1) & 0x3;
            acc = fpmac(acc, data, 0, 0x76543210, wvec, 0, 0x00000000);
            acc = fpmac(acc, data, 1, 0x76543210, wvec, 1, 0x00000000);
            acc = fpmac(acc, data, 2, 0x76543210, wvec, 2, 0x00000000);
            acc = fpmac(acc, data, 3, 0x76543210, wvec, 3, 0x00000000);
            acc = fpmac(acc, data, 4, 0x76543210, wvec, 4, 0x00000000);

            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            GET_WVEC(wp, zstart);
            wp += 5;
            zstart = (zstart + 1) & 0x3;
            acc = fpmac(acc, data, 0, 0x76543210, wvec, 0, 0x00000000);
            acc = fpmac(acc, data, 1, 0x76543210, wvec, 1, 0x00000000);
            acc = fpmac(acc, data, 2, 0x76543210, wvec, 2, 0x00000000);
            acc = fpmac(acc, data, 3, 0x76543210, wvec, 3, 0x00000000);
            acc = fpmac(acc, data, 4, 0x76543210, wvec, 4, 0x00000000);

            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            GET_WVEC(wp, zstart);
            wp += 5;
            zstart = (zstart + 1) & 0x3;
            acc = fpmac(acc, data, 0, 0x76543210, wvec, 0, 0x00000000);
            acc = fpmac(acc, data, 1, 0x76543210, wvec, 1, 0x00000000);
            acc = fpmac(acc, data, 2, 0x76543210, wvec, 2, 0x00000000);
            acc = fpmac(acc, data, 3, 0x76543210, wvec, 3, 0x00000000);
            acc = fpmac(acc, data, 4, 0x76543210, wvec, 4, 0x00000000);

            data = upd_w(data, 0, window_readincr_v8(in));
            data = upd_w(data, 1, window_readincr_v8(in));
            window_incr(in, INP_W-16);
            GET_WVEC(wp, zstart);
            wp += 5;
            zstart = (zstart + 1) & 0x3;
            acc = fpmac(acc, data, 0, 0x76543210, wvec, 0, 0x00000000);
            acc = fpmac(acc, data, 1, 0x76543210, wvec, 1, 0x00000000);
            acc = fpmac(acc, data, 2, 0x76543210, wvec, 2, 0x00000000);
            acc = fpmac(acc, data, 3, 0x76543210, wvec, 3, 0x00000000);
            acc = fpmac(acc, data, 4, 0x76543210, wvec, 4, 0x00000000);

            window_incr(in, INP_W*INP_W - 5*INP_W);
          }
          window_incr(in, -C*INP_W*INP_W + 8); // data go channel -C, right 8
                    
          acc = fpmax(acc, zeros, 0, 0x76543210);
          int outincr = (OUT_W - w < 8 && OUT_W - w > 0) ? OUT_W - w : 8;
          window_write(out, acc);
          window_incr(out, outincr);

        } // W
        window_incr(in, INP_W-OUT_W/8*8); // go left OUT_W/8*8, go down 1

      } // H

      window_incr(in, -OUT_W*INP_W); // go up OUT_W
    } // M
  } // B

  PROFILE_FOOTER;
}