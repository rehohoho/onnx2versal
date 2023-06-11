#include "qlinearmac.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"



template <int B, int W, int IS_RELU>
QlinearMacScalar<B, W, IS_RELU>::QlinearMacScalar (
  int8_t (&w)[W],
  int8_t (&b)[W],
  float x_scale,
  float w_scale,
  float b_scale,
  float z_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t b_zero,
  int8_t z_zero,
  int8_t y_zero
): 
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), b_scale(b_scale), z_scale(z_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), b_zero(b_zero), z_zero(z_zero), y_zero(y_zero) 
{
  scale_x = x_scale * w_scale * inv(z_scale);
  scale_z = z_scale * inv(y_scale);
  for (int w = 0; w < W; w++) {
    shift_x[w] = -x_zero * (weights[w] - w_zero) * scale_x + z_zero;
    shift_z[w] = (-z_zero * z_scale + (bias[w] - b_zero) * b_scale) * inv(y_scale) + y_zero;
  }
}

template <int B, int W, int IS_RELU>
void QlinearMacScalar<B, W, IS_RELU>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QlinearMacScalar<%d,%d,%d>\n", B, W, IS_RELU));
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w++) {
      int x = window_readincr(in);
      x = x * (weights[w] - w_zero);
      x = round(x * scale_x + shift_x[w]);
      x = std::min(std::max(x, -128), 127);
      // printf("%d ", x);

      int y = round(x * scale_z + shift_z[w]);
      y = std::min(std::max(y, -128), 127);
      if (IS_RELU)
        y = (y > 0) ? y : 0;
      window_writeincr(out, y);
    }
    // printf("\n");
  }

  PROFILE_FOOTER;
}


template <int B, int W, int IS_RELU>
QlinearMac<B, W, IS_RELU>::QlinearMac (
  int8_t (&w)[W],
  int8_t (&b)[W],
  float x_scale,
  float w_scale,
  float b_scale,
  float z_scale,
  float y_scale,
  int8_t x_zero,
  int8_t w_zero,
  int8_t b_zero,
  int8_t z_zero,
  int8_t y_zero
): 
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), b_scale(b_scale), z_scale(z_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), b_zero(b_zero), z_zero(z_zero), y_zero(y_zero) 
{
  assert(w_zero == 0);
  float invlog2 = inv(log(2));

  float fscale_x = x_scale * w_scale * inv(z_scale);
  bitshift_x = std::abs(log(fscale_x) * invlog2) + 18;
  scale_x = float2fix(fscale_x, bitshift_x);

  float fscale_z = z_scale * inv(y_scale);
  bitshift_z = std::abs(log(fscale_z) * invlog2) + 18;
  scale_z = float2fix(fscale_z, bitshift_z);

  for (int w = 0; w < W; w++) {
    float fshift_x = -x_zero * (weights[w] - w_zero) * fscale_x + z_zero;
    shift_x[w] = float2fix(fshift_x, bitshift_x);
    float fshift_z = (-z_zero * z_scale + (bias[w] - b_zero) * b_scale) * inv(y_scale) + y_zero;
    shift_z[w] = float2fix(fshift_z, bitshift_z);
  }
  printf("bitshift_x %d bitshift_z %d\n", bitshift_x, bitshift_z);
}

template <int B, int W, int IS_RELU>
void QlinearMac<B, W, IS_RELU>::filter(
	input_window<int8_t>* in,
  output_window<int8_t>* out
) {
  PROFILE_HEADER(printf(
    "Running QlinearMac<%d,%d,%d>\n", B, W, IS_RELU));
  
  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  aie::accum<acc48,16> acc1;
  aie::accum<acc48,16> acc2;

  int8_t *out_ptr = (int8_t *) out->ptr;
  int8_t *w_ptr = (int8_t *) weights;
  int32_t *sx_ptr = (int32_t *) shift_x;
  int32_t *sz_ptr = (int32_t *) shift_z;
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w+=16) {
      
      aie::vector<int8_t,16> x = window_readincr_v16(in);      
      aie::vector<int8_t,16> wv = aie::load_v<16>(w_ptr); w_ptr += 16;
      auto z = aie::mul(x, wv).to_vector<int16_t>(0);
      
      acc1.from_vector(aie::load_v<16>(sx_ptr), 0); sx_ptr += 16;
      auto zz = aie::mac(acc1, z, scale_x).to_vector<int16_t>(bitshift_x);

      // print_vec<short, short>((short *) &z, 16);
      
      acc2.from_vector(aie::load_v<16>(sz_ptr), 0); sz_ptr += 16;
      auto res = aie::mac(acc2, zz, scale_z).to_vector<int8_t>(bitshift_z);
      
      if (IS_RELU)
        res = aie::max(res, (int8_t) 0);
      
      aie::store_v(out_ptr, res); out_ptr += 16;
    }
    // printf("\n");
    w_ptr -= W;
    sx_ptr -= W;
    sz_ptr -= W;
  }

  PROFILE_FOOTER;
}
