#include "pad.h"
#include "kernel_utils.h"
#include "aie_api/aie.hpp"


#define PAD_PROFILE_FOOTER(filter_name) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d,%d,%d,%d,%d>", \
    filter_name, typeid(TT).name(), B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1);


template <typename TT, int B, int INP_H, int INP_W, int INP_W_PAD, int H0, int H1, int W0, int W1>
void Pad2DStreamInt8<TT, B, INP_H, INP_W, INP_W_PAD, H0, H1, W0, W1>::filter(
	input_stream<TT>* restrict in,
  output_stream<TT>* restrict out
) {
  PROFILE_HEADER2;
  
  using TTVEC = typename std::conditional<(std::is_same<TT, int8_t>::value), v16int8, v16uint8>::type;
  
  v64int16 data = aie::broadcast<int16_t,64>(pad_value);
  TTVEC pad_value_v = aie::broadcast<TT,16>(pad_value);
  v16int16 pad_value_vint16 = aie::broadcast<int16_t,16>(pad_value);
  int data_offset = 0;

  // for top and bottom pads, assume OUT_W >= 16
#define WRITE_PAD(out, len) \
  data = aie::shuffle_down((aie::vector<int16_t,64>) data, data_offset); \
  data = upd_w(data, 3, pad_value_vint16); \
  if (data_offset + len >= 16) { \
    data = aie::shuffle_down_replicate((aie::vector<int16_t,64>) data, 16-data_offset); \
    writeincr_v16(out, ((aie::vector<int16,16>) ext_w(data, 2)).pack<TT>()); \
    for (int i = 0; i <= (len - (16 - data_offset)) - 16; i+=16) \
      writeincr_v16(out, pad_value_v); \
  } else { \
    data = aie::shuffle_up((aie::vector<int16_t,64>) data, data_offset); \
  } \
  data_offset = (data_offset + len) & 0xf;

  for (int b = 0; b < B; b++) chess_prepare_for_pipelining chess_loop_range(B,) {
    WRITE_PAD(out, H0*OUT_W+W0);
    
    for (int h = 0; h < INP_H; h++) chess_prepare_for_pipelining chess_loop_range(INP_W,) {
      // main1
      for (int w = 0; w < INP_W/48; w++) {
        // shuffle remaining data to end, update front, shuffle rotate so data starts with remaining data
        data = aie::shuffle_up((aie::vector<int16_t,64>) data, 16 - data_offset);
        data = upd_w(data, 0, unpack(getb_wss(0)));
        data = upd_w(data, 1, unpack(getb_wss(0)));
        data = upd_w(data, 2, unpack(getb_wss(0)));
        data = aie::shuffle_up((aie::vector<int16_t,64>) data, data_offset);
        writeincr_v16(out, ((aie::vector<int16,16>) ext_w(data, 0)).pack<TT>());
        writeincr_v16(out, ((aie::vector<int16,16>) ext_w(data, 1)).pack<TT>());
        writeincr_v16(out, ((aie::vector<int16,16>) ext_w(data, 2)).pack<TT>());
      }
      if (INP_W % 48 > 0) {
        data = aie::shuffle_up((aie::vector<int16_t,64>) data, 16 - data_offset);
        for (int i = 0; i < (INP_W % 48) / 16; i++) {
          data = upd_w(data, i, unpack(getb_wss(0)));
        }
        data = aie::shuffle_up((aie::vector<int16_t,64>) data, data_offset);
        for (int i = 0; i < (INP_W % 48) / 16; i++) {
          writeincr_v16(out, ((aie::vector<int16,16>) ext_w(data, i)).pack<TT>());
        }
        data = aie::shuffle_up((aie::vector<int16_t,64>) data, (3 - (INP_W % 48)/16) * 16);
      }
      
      if (INP_W % 16 != 0) {
        data = aie::shuffle_down((aie::vector<int16_t,64>) data, data_offset);
        data = upd_w(data, 3, unpack(getb_wss(0)));
        if (data_offset + INP_W % 16 >= 16) {
          data = aie::shuffle_down_replicate((aie::vector<int16_t,64>) data, 16-data_offset);
          writeincr_v16(out, ((aie::vector<int16,16>) ext_w(data, 2)).pack<TT>());
          data_offset -= 16;
        } else {
          data = aie::shuffle_up((aie::vector<int16_t,64>) data, data_offset);
        }
        data_offset += INP_W % 16;
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
