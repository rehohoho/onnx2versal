#include "qlinearmac.h"
#include "kernel_utils.h"


#define QLINEARMAC_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%s,%d,%d,%d>", \
    filter_name, typeid(TT).name(), typeid(TTPARAM).name(), B, W, IS_RELU);


template <typename TT, typename TTPARAM, int B, int W, int IS_RELU>
QlinearMacScalar<TT, TTPARAM, B, W, IS_RELU>::QlinearMacScalar (
  TTPARAM (&w)[W],
  TTPARAM (&b)[W],
  float x_scale,
  float w_scale,
  float b_scale,
  float z_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TTPARAM b_zero,
  TT z_zero,
  TT y_zero
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

template <typename TT, typename TTPARAM, int B, int W, int IS_RELU>
void QlinearMacScalar<TT, TTPARAM, B, W, IS_RELU>::filter(
	input_window<TT>* in,
  output_window<TT>* out
) {
  PROFILE_HEADER2;
  
  for (int b = 0; b < B; b++) {
    for (int w = 0; w < W; w++) {
      int x = window_readincr(in);
      x = x * weights[w];
      x = round(x * scale_x + shift_x[w]);
      if ((std::is_same<TT, int8_t>::value)) {
        x = std::min(std::max(x, -128), 127);
      } else {
        x = std::min(std::max(x, 0), 255);
      }

      int y = round(x * scale_z + shift_z[w]);
      if ((std::is_same<TT, int8_t>::value)) {
        y = std::min(std::max(y, -128), 127);
      } else {
        y = std::min(std::max(y, 0), 255);
      }
      if (IS_RELU)
        y = (y > 0) ? y : 0;
      window_writeincr(out, y);
    }
    // printf("\n");
  }

  QLINEARMAC_PROFILE_FOOTER("QlinearMacScalar");
}


template <typename TT, typename TTPARAM, int B, int W, int IS_RELU>
QlinearMac<TT, TTPARAM, B, W, IS_RELU>::QlinearMac (
  TTPARAM (&w)[W],
  TTPARAM (&b)[W],
  float x_scale,
  float w_scale,
  float b_scale,
  float z_scale,
  float y_scale,
  TT x_zero,
  TTPARAM w_zero,
  TTPARAM b_zero,
  TT z_zero,
  TT y_zero
): 
  weights(w), bias(b), 
  x_scale(x_scale), w_scale(w_scale), b_scale(b_scale), z_scale(z_scale), y_scale(y_scale), 
  x_zero(x_zero), w_zero(w_zero), b_zero(b_zero), z_zero(z_zero), y_zero(y_zero) 
{
  assert(w_zero == 0);

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  float fscale_x = x_scale * w_scale / z_scale;
  float fscale_z = z_scale / y_scale;
  bitshift_x = 15 - log(fscale_x) / log(2); // fshift_x uses int8*int8*fscale_x+int8 <= int32
  bitshift_z = 15 - log(fscale_z) / log(2);
  scale_z = float2fix(fscale_z, bitshift_z);

  float fscale_b = b_scale / y_scale;
  
  for (int w = 0; w < W; w++) {
    float fscale_w = weights[w]*fscale_x;
    float fshift_x = -x_zero*fscale_x * weights[w] + z_zero;
    float fshift_z = -z_zero*fscale_z + (bias[w]-b_zero)*fscale_b + y_zero;
    scale_x[w] = float2fix(fscale_w, bitshift_x);
    shift_x[w] = float2fix(fshift_x, bitshift_x);
    shift_z[w] = float2fix(fshift_z, bitshift_z);
    // printf("scale_x=%d shift_x=%d shift_z=%d\n", scale_x[w], shift_x[w], shift_z[w]);
  }
  // printf("\nbitshift_x %d bitshift_z %d scale_z %d\n", bitshift_x, bitshift_z, scale_z);
}

template <typename TT, typename TTPARAM, int B, int W, int IS_RELU>
void QlinearMac<TT, TTPARAM, B, W, IS_RELU>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  using v16 = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  v16acc48 acc1 = null_v16acc48();
  aie::accum<acc48,16> accx;

  v16 res1 = aie::zeros<TT,16>();
  v16int32 *w_ptr = (v16int32 *) scale_x;

  int32_t *sx_ptr = (int32_t *) shift_x;
  int32_t *sz_ptr = (int32_t *) shift_z;
  
  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    
    for (int w = 0; w < W; w+=16) chess_prepare_for_pipelining chess_loop_range(W/16,) {
      accx.from_vector(aie::load_v<16>(sx_ptr), 0); sx_ptr += 16;
      acc1 = aie::mac(accx, (aie::vector<int16_t,16>) unpack(readincr_v16(in)), (aie::vector<int32_t,16>) *w_ptr); w_ptr++;
      
      accx.from_vector(aie::load_v<16>(sz_ptr), 0); sz_ptr += 16;
      acc1 = aie::mac(accx, ((aie::accum<acc48,16>) acc1).to_vector<TT>(bitshift_x), scale_z);
      res1 = ((aie::accum<acc48,16>) acc1).to_vector<TT>(bitshift_z);
      
      if (IS_RELU) {
        res1 = aie::max((aie::vector<TT,16>) res1, (TT) 0);
      }
      
      writeincr_v16(out, res1);
    }
    sx_ptr -= W;
    sz_ptr -= W;
    w_ptr -= W/16;

  }

  QLINEARMAC_PROFILE_FOOTER("QlinearMac");
}
