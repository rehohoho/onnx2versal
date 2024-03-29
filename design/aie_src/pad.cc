#include "pad.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define PAD_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1);


// bandwidth limited, not much speedup
template <typename TT, int B, int INP_H, int INP_W, int INP_W_PAD, int H0, int H1, int W0, int W1>
void Pad2DStreamFloat<TT, B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>::filter(
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

      for (int w = 0; w <= INP_W-4; w+=4) {
        a = getf_wss(0);
        put_wms(0, a);
      }
      for (int w = 0; w < INP_W % 4; w++)
        put_ms(0, get_ss(0));
      for (int w = 0; w < INP_W_PAD - INP_W; w++)
        get_ss(0);

      WRITE_PAD(out, W1);
    }

    WRITE_PAD(out, H1*OUT_W);
  }

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DStreamFloat");
}


// slight reg spilling, using v64int16 will be worse
// shuffle ops requires minimum int16 size, aie::shuffle for int8/uint8 unpacks first and repacks
template <typename TT, int B, int INP_H, int INP_W, int INP_W_PAD, int H0, int H1, int W0, int W1>
void Pad2DStreamInt8<TT, B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  using TTVEC = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  
  aie::vector<int16_t, 32> data = aie::broadcast<int16_t,32>(pad_value);
  aie::vector<int16_t, 16> tmp;
  int data_offset = 0;

  // for top and bottom pads, assume OUT_W >= 16
#define WRITE_PAD(out, len) \
  data = aie::shuffle_down(data, data_offset); \
  tmp = aie::broadcast<int16_t,16>(pad_value); \
  data = data.insert(1, tmp); \
  if (data_offset + len >= 16) { \
    data = aie::shuffle_down_replicate(data, 16-data_offset); \
    writeincr_v16(out, data.extract<16>(0).pack<TT>()); \
    for (int i = 0; i <= (len - (16 - data_offset)) - 16; i+=16) \
      writeincr_v16(out, aie::broadcast<TT,16>(pad_value)); \
  } else { \
    data = aie::shuffle_up(data, data_offset); \
  } \
  data_offset = (data_offset + len) & 0xf;

  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    WRITE_PAD(out, H0*OUT_W+W0);
    
    for (int h = 0; h < INP_H; h++) chess_prepare_for_pipelining chess_loop_range(INP_W,) {
      for (int w = 0; w <= INP_W-16; w+=16) {
        // shuffle remaining data to end, update front, shuffle rotate so data starts with remaining data
        data = aie::shuffle_up(data, 16 - data_offset);
        tmp = readincr_v<16>(in).unpack();
        data = data.insert(0, tmp);
        data = aie::shuffle_up(data, data_offset);
        writeincr_v16(out, data.extract<16>(0).pack<TT>());
      }
      
      if ((INP_W & 0xf) != 0) {
        data = aie::shuffle_down(data, data_offset);
        tmp = unpack(getb_wss(0));
        data = data.insert(1, tmp);
        if (data_offset + (INP_W & 0xf) >= 16) {
          data = aie::shuffle_down_replicate(data, 16-data_offset);
          writeincr_v16(out, data.extract<16>(0).pack<TT>());
          data_offset -= 16;
        } else {
          data = aie::shuffle_up(data, data_offset);
        }
        data_offset += (INP_W & 0xf);
      }
      
      WRITE_PAD(out, W0+W1);
    }
    WRITE_PAD(out, H1*OUT_W-W0);
    chess_separator_scheduler();
  }

  for (int i = 0; i < data_offset; i+=4)
    put_ms(0, 0);

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DStreamInt8");
}


// stream int8 requires shuffle since bitwidth=32 or 128
template <typename TT, int B, int INP_H, int INP_W, int INP_W_PAD, int H0, int H1, int W0, int W1>
void Pad2DWindowScalar<TT, B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>::filter(
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
    window_incr(in, INP_W_PAD - INP_W);

    WRITE_PAD(out, W0+W1);
  }

  WRITE_PAD(out, H1*OUT_W - W0);

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DWindowScalar");
}


template <typename TT, int B, int INP_H, int INP_W, int INP_W_PAD, int H0, int H1, int W0, int W1>
void Pad2DStream2WindowInt8<TT, B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_window<TT>* restrict out
) {
  PROFILE_HEADER2;

  TT* out_ptr = (TT *) out->ptr;
  aie::vector<TT,16> pad_value_vector = aie::broadcast<TT, 16>(pad_value);

#define WRITE_PAD(out, len) \
  if (pad_value == 0) { \
    out_ptr += len; \
  } else { \
    int len_padded = (len + 15)/16*16; \
    for (int i = 0; i < len_padded; i+=16) \
      aie::store_unaligned_v<16>(out_ptr, pad_value_vector); out_ptr += 16; \
    out_ptr -= len_padded - len; \
  }

  WRITE_PAD(out, H0*OUT_W + W0);
  
  for (int h = 0; h < INP_H; h++) {

    for (int w = 0; w < INP_W_PAD; w+=16) {
      aie::vector<TT,16> data = readincr_v16(in);
      aie::store_unaligned_v<16>(out_ptr, data); out_ptr += 16;
    }
    out_ptr -= INP_W_PAD - INP_W;

    WRITE_PAD(out, W0+W1);
  }

  WRITE_PAD(out, H1*OUT_W - W0);

#undef WRITE_PAD

  PAD_PROFILE_FOOTER("Pad2DStream2WindowInt8");
}