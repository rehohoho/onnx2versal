#include "pad.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define PAD_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, INP_H, INP_W, H0, H1, W0, W1);

template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DStreamScalar<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;

#define WRITE_PAD(out, len) \
  for (int i = 0; i < len; i++) writeincr(out, pad_value);

  for (int b = 0; b < B; b++) {
    WRITE_PAD(out, H0*OUT_W);
    
    for (int h = 0; h < INP_H; h++) {
      WRITE_PAD(out, W0);

      for (int w = 0; w < INP_W; w++) {
        writeincr(out, readincr(in));
      }

      WRITE_PAD(out, W1);
    }

    WRITE_PAD(out, H1*OUT_W);
  }

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DStreamScalar");
}


// bandwidth limited, not much speedup
template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DStreamFloat<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  v4float a = undef_v4float();
  v4float zero = aie::broadcast<float, 4>(pad_value);

#define WRITE_PAD(out, len) \
  for (int i = 0; i <= len-4; i+=4) put_wms(0, zero); \
  for (int i = 0; i < len%4; i++) put_ms(0, pad_value);

  for (int b = 0; b < B; b++) {
    WRITE_PAD(out, H0*OUT_W);
    
    for (int h = 0; h < INP_H; h++) {
      WRITE_PAD(out, W0);

      for (int w = 0; w < INP_W; w+=4) {
        a = getf_wss(0);
        put_wms(0, a);
      }

      WRITE_PAD(out, W1);
    }

    WRITE_PAD(out, H1*OUT_W);
  }

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DStreamFloat");
}


template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DStreamInt8<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  v32int16 data = aie::broadcast<int16_t,32>(pad_value);
  v16int8 pad_value_v = aie::broadcast<int8_t,16>(pad_value);
  v16int16 pad_value_vint16 = aie::broadcast<int16_t,16>(pad_value);
  int data_offset = 0;

  // for top and bottom pads, assume OUT_W >= 16
#define WRITE_PAD(out, len) \
  data = aie::shuffle_down((aie::vector<int16_t,32>) data, data_offset); \
  data = upd_w(data, 1, pad_value_vint16); \
  if (data_offset + len >= 16) { \
    data = aie::shuffle_down_replicate((aie::vector<int16_t,32>) data, 16-data_offset); \
    put_wms(0, pack(ext_w(data, 0))); \
    for (int i = 0; i <= (len - (16 - data_offset)) - 16; i+=16) \
      put_wms(0, pad_value_v); \
  } else { \
    data = aie::shuffle_up((aie::vector<int16_t,32>) data, data_offset); \
  } \
  data_offset = (data_offset + len) & 0xf;

  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    WRITE_PAD(out, H0*OUT_W+W0);
    
    for (int h = 0; h < INP_H; h++) chess_prepare_for_pipelining chess_loop_range(INP_W,) {
      for (int w = 0; w < INP_W; w+=16) chess_prepare_for_pipelining chess_loop_range(INP_W/16,) {
        // shuffle remaining data to end, update front, shuffle rotate so data starts with remaining data
        data = aie::shuffle_up((aie::vector<int16_t,32>) data, 16 - data_offset);
        data = upd_w(data, 0, unpack(getb_wss(0)));
        data = aie::shuffle_up((aie::vector<int16_t,32>) data, data_offset);
        put_wms(0, pack(ext_w(data, 0))); 
      }
      WRITE_PAD(out, W0+W1);
    }
    WRITE_PAD(out, H1*OUT_W-W0);
  }

  for (int i = 0; i < data_offset; i+=4)
    put_ms(0, 0);

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DStreamInt8");
}


// stream int8 requires shuffle since bitwidth=32 or 128
template <typename TT, int B, int INP_H, int INP_W, int H0, int H1, int W0, int W1>
void Pad2DWindowScalar<TT, B, INP_H, INP_W, H0, H1, W0, W1>::filter(
	input_window<TT>* restrict in,
  output_window<TT>* restrict out
) {
  PROFILE_HEADER2;

#define WRITE_PAD(out, len) \
  if (pad_value == 0) { \
    window_incr(out, len); \
  } else { \
    for (int i = 0; i < len; i++) \
      window_writeincr(out, pad_value); \
  }

  WRITE_PAD(out, H0*OUT_W + W0);
  
  for (int h = 0; h < INP_H; h++) {

    for (int w = 0; w < INP_W; w++) {
      TT a = window_readincr(in);
      window_writeincr(out, a);
    }

    WRITE_PAD(out, W0+W1);
  }

  WRITE_PAD(out, H1*OUT_W - W0);

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DWindowScalar");
}