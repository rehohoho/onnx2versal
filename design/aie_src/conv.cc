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
               int  	        zstart,
               unsigned int  	zoffs)

for (i = 0; i < 8; i++)
  ret[i] = acc[i] + xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]]

Reuses weights.

8 outputs per loop:
in[(k*row)+i:(k*row)+i+8] * weights[k*K+i], 0<=i<=K

*/
#define print_vecs \
  if (c == 0) { \
    float* print_x = (float*) &data; \
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n", \
      print_x[0], print_x[1], print_x[2], print_x[3], print_x[4], print_x[5], print_x[6], print_x[7], \
      print_x[8], print_x[9], print_x[10], print_x[11], print_x[12], print_x[13], print_x[14], print_x[15]); \
    float* print_w = (float*) &wvec; \
    printf("%f %f %f %f %f %f %f %f \n", print_w[0], print_w[1], print_w[2], print_w[3], print_w[4], print_w[5], print_w[6], print_w[7]); \
  }
#define conv_krow(ZSTART) \
  wvec = *(v8float*) (weights+widx); \
  data = upd_x(data, 0, window_read_v16(in)); \
  acc = fpmac(acc, data, 0, 0x76543210, wvec, ZSTART, 0x00000000); \
  acc = fpmac(acc, data, 1, 0x76543210, wvec, ZSTART, 0x11111111); \
  acc = fpmac(acc, data, 2, 0x76543210, wvec, ZSTART, 0x22222222); \
  acc = fpmac(acc, data, 3, 0x76543210, wvec, ZSTART, 0x33333333); \
  acc = fpmac(acc, data, 4, 0x76543210, wvec, ZSTART, 0x44444444);

template <int INP_W, int OUT_W, int B, int C, int M>
void Conv5x5ReluBCHW<INP_W, OUT_W, B, C, M>::filter(
	input_window<float>* in,      // BCHW
  output_window<float>* out     // BMHW
) {
  PROFILE_HEADER;
  printf("Running Conv5x5ReluBCHW<%d, %d, %d, %d, %d>\n", INP_W, OUT_W, B, C, M);

  v32float data = null_v32float();
  v8float zeros = null_v8float();
  v8float wvec = undef_v8float();
  float* wp; // MCKK
  int widx, zstart;
  
  // BHWM
  for (int b = 0; b < B; b++) {
    for (int m = 0; m < M; m++) { // computes one output channel
      for (int h = 0; h < OUT_W; h++) {
        for (int w = 0; w < OUT_W; w+=8) { // computes 8 output channel pixels
          
          v8float acc = aie::broadcast<float, 8>(bias[m]);
          widx = (m*C*5*5)/4*4;
          zstart = m*C*5*5 & 0x3;

          for (int c = 0; c < C; c++) {   // computes 8 partial products
            // load in wvec[0:8], use 0:5
            conv_krow(zstart);
            zstart = (zstart + 1) & 0x3;
            widx += ((widx + 8) % 20 == 0) ? 8 : 4;
            window_incr(in, INP_W);

            // load in wvec[4:12] use 5:10
            conv_krow(zstart);
            zstart = (zstart + 1) & 0x3;
            widx += ((widx + 8) % 20 == 0) ? 8 : 4; 
            window_incr(in, INP_W);

            // load in wvec[8:16] use 10:15
            conv_krow(zstart);
            zstart = (zstart + 1) & 0x3;
            widx += ((widx + 8) % 20 == 0) ? 8 : 4; 
            window_incr(in, INP_W);

            // load in wvec[12:20] use 15:20
            conv_krow(zstart);
            zstart = (zstart + 1) & 0x3;
            widx += ((widx + 8) % 20 == 0) ? 8 : 4; 
            window_incr(in, INP_W);

            // load in wvec[20:28] use 20:25
            conv_krow(zstart);
            zstart = (zstart + 1) & 0x3;
            widx += ((widx + 8) % 20 == 0) ? 8 : 4; 
            window_incr(in, INP_W*INP_W - 4*INP_W); // data go up 4, channel 1
          }
          // printf("\n");

          window_incr(in, -C*INP_W*INP_W + 8); // data go channel -C, right 8
          
          acc = fpmax(acc, zeros, 0, 0x76543210);
          int outincr = (OUT_W - w < 8 && OUT_W - w > 0) ? OUT_W - w : 8;
          window_write(out, acc);
          window_incr(out, outincr);

        } // W
        window_incr(in, INP_W-OUT_W/8*8); // go left OUT_W/8*8, go down 1
        // printf("\n");
      } // H
      window_incr(in, -OUT_W*INP_W); // go up OUT_W
      // printf("\n");
    } // M
  } // B

  PROFILE_FOOTER;
}
