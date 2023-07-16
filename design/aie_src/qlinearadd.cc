#include "qlinearadd.h"
#include "kernel_utils.h"


#define QLINEARADD_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d>", \
    filter_name, typeid(TT).name(), W, IS_RELU);

template <typename TT, int W, int IS_RELU>
QLinearAddInt8<TT, W, IS_RELU>::QLinearAddInt8 (
  float a_scale,
  float b_scale,
  float c_scale,
  TT a_zero,
  TT b_zero,
  TT c_zero
): a_scale(a_scale), b_scale(b_scale), c_scale(c_scale), a_zero(a_zero), b_zero(b_zero), c_zero(c_zero) {
  float invc = inv(c_scale);
  bitshift = 16;
  ascale = float2fix(a_scale * invc, bitshift);
  bscale = float2fix(b_scale * invc, bitshift);
  shiftv = float2fix((-a_zero*a_scale -b_zero*b_scale) * invc + c_zero, bitshift);
  // printf("bitshift %d ascale %d bscale %d shiftv %d\n", bitshift, ascale, bscale, shiftv);
};

// int32_t factors for precision over accuracy
template <typename TT, int W, int IS_RELU>
void QLinearAddInt8<TT, W, IS_RELU>::filter(
	input_stream<TT>* restrict inA,
  input_stream<TT>* restrict inB,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

  set_sat();
  set_rnd(rnd_sym_inf); // c++: round halfway towards infinity, away from zero

  aie::vector<int16_t, 16> a;
  aie::vector<int16_t, 16> b;
  aie::vector<TT, 16> res;
  
  aie::accum<acc48, 16> _a;
  aie::accum<acc48, 16> shift;
  shift.from_vector(aie::broadcast<int32_t, 16>(shiftv), 0);

  for (int w = 0; w < W; w+=16) {
    a = unpack(readincr_v16(inA));
    b = unpack(readincr_v16(inB));

    _a = aie::mac(shift, a, ascale);
    _a = aie::mac(_a, b, bscale);
    res = _a.to_vector<TT>(bitshift);

    if (IS_RELU)
      res = aie::max(res, (TT) 0);

    writeincr_v16(out, res);
  }

  QLINEARADD_PROFILE_FOOTER("QLinearAddInt8");
}
