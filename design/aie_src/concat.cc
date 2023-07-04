#include "concat.h"
#include "kernel_utils.h"


#define CONCAT_PROFILE_FOOTER(filter_name, filter_i) \
  PROFILE_FOOTER2("%s<%s,%d,%d,%d,%d>::filter%d", \
    filter_name, typeid(TT).name(), LCNT, H, INP_W, OUT_W, filter_i);

#define CAT(INP_WIN) \
  if (outi + INP_W <= OUT_W) { \
    for (int i = 0; i < INP_W; i++) \
      writeincr(out, window_readincr(INP_WIN)); \
  } else if (outi < OUT_W) { \
    for (int i = 0; i < OUT_W - outi; i++) \
      writeincr(out, window_readincr(INP_WIN)); \
    window_incr(INP_WIN, INP_W - OUT_W + outi); \
  } \
  outi += INP_W;


template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter8(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
	input_window<TT>* in6,
	input_window<TT>* in7,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    CAT(in6);
    CAT(in7);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 8);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter7(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
	input_window<TT>* in6,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    CAT(in6);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 7);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter6(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    CAT(in5);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 6);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter5(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    CAT(in4);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 5);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter4(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    CAT(in3);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 4);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter3(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 3);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter2(
	input_window<TT>* in0,
	input_window<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 2);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter1(
	input_window<TT>* in0,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT(in0);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatScalar", 1);
}


#define CAT_FLOAT(INP_WIN) \
  if (outi + INP_W <= OUT_W) { \
    for (int i = 0; i < INP_W; i+=4) \
      writeincr_v4(out, window_readincr_v4(INP_WIN)); \
  } else if (outi < OUT_W) { \
    for (int i = 0; i <= OUT_W - outi - 4; i+=4) \
      writeincr_v4(out, window_readincr_v4(INP_WIN)); \
    for (int i = 0; i < (OUT_W - outi) % 4; i++) \
      writeincr(out, window_readincr(INP_WIN)); \
    window_incr(INP_WIN, outi + INP_W - OUT_W); \
  } \
  outi += INP_W;

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter8(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
	input_window<TT>* in6,
	input_window<TT>* in7,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    CAT_FLOAT(in3);
    CAT_FLOAT(in4);
    CAT_FLOAT(in5);
    CAT_FLOAT(in6);
    CAT_FLOAT(in7);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 8);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter7(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
	input_window<TT>* in6,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    CAT_FLOAT(in3);
    CAT_FLOAT(in4);
    CAT_FLOAT(in5);
    CAT_FLOAT(in6);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 7);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter6(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    CAT_FLOAT(in3);
    CAT_FLOAT(in4);
    CAT_FLOAT(in5);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 6);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter5(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    CAT_FLOAT(in3);
    CAT_FLOAT(in4);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 5);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter4(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    CAT_FLOAT(in3);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 4);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter3(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 3);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter2(
	input_window<TT>* in0,
	input_window<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 2);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter1(
	input_window<TT>* in0,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_FLOAT(in0);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatFloat", 1);
}


// assumes INP_W%8=0, OUT_W%4=0 (vector writes)
#define CAT_INT8(INP_WIN) \
  if (outi + INP_W <= OUT_W) { \
    for (int i = 0; i < INP_W; i+=16) \
      writeincr_v16(out, window_readincr_v16(INP_WIN)); \
  } else if (outi < OUT_W) { \
    for (int i = 0; i < OUT_W - outi - 15; i+=16) \
      writeincr_v16(out, window_readincr_v16(INP_WIN)); \
    for (int i = 0; i < (OUT_W - outi) % 16; i++) \
      writeincr(out, window_readincr(INP_WIN)); \
    window_incr(INP_WIN, outi + INP_W - OUT_W); \
  } \
  outi += INP_W;

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter8(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
	input_window<TT>* in6,
	input_window<TT>* in7,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    CAT_INT8(in3);
    CAT_INT8(in4);
    CAT_INT8(in5);
    CAT_INT8(in6);
    CAT_INT8(in7);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 8);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter7(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
	input_window<TT>* in6,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    CAT_INT8(in3);
    CAT_INT8(in4);
    CAT_INT8(in5);
    CAT_INT8(in6);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 7);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter6(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
	input_window<TT>* in5,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    CAT_INT8(in3);
    CAT_INT8(in4);
    CAT_INT8(in5);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 6);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter5(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
	input_window<TT>* in4,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    CAT_INT8(in3);
    CAT_INT8(in4);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 5);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter4(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    CAT_INT8(in3);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 4);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter3(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 3);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter2(
	input_window<TT>* in0,
	input_window<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 2);
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter1(
	input_window<TT>* in0,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    int outi = 0;
    CAT_INT8(in0);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  CONCAT_PROFILE_FOOTER("ConcatInt8", 1);
}


template <typename TT, int H, int INP_W1, int INP_W2, int OUT_W>
void ConcatFloatStream<TT, H, INP_W1, INP_W2, OUT_W>::filter(
	input_stream<TT>* in0,
  input_stream<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    for (int i = 0; i < INP_W1; i++)
      writeincr(out, readincr(in0));
    
    if (INP_W1 + INP_W2 <= OUT_W) {
      for (int i = 0; i < INP_W2; i++)
        writeincr(out, readincr(in1));
      for (int i = 0; i < OUT_W - INP_W1 - INP_W2; i++)
        writeincr(out, 0);
    } else {
      for (int i = 0; i < OUT_W - INP_W1; i++)
        writeincr(out, readincr(in1));  
      for (int i = 0; i < INP_W1 + INP_W2 - OUT_W; i++)
        readincr(in1);
    }
  }

  PROFILE_FOOTER2("ConcatFloatStream<%s,%d,%d,%d,%d>", 
    typeid(TT).name(), H, INP_W1, INP_W2, OUT_W);
}


template <typename TT, int H, int INP_W1, int INP_W2, int OUT_W>
void ConcatInt8Stream<TT, H, INP_W1, INP_W2, OUT_W>::filter(
	input_stream<TT>* in0,
  input_stream<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  for (int h = 0; h < H; h++) {
    for (int i = 0; i < INP_W1; i+=16)
      writeincr_v16(out, readincr_v16(in0));
    
    if (INP_W1 + INP_W2 <= OUT_W) {
      for (int i = 0; i < INP_W2; i+=16)
        writeincr_v16(out, readincr_v16(in1));
      for (int i = 0; i < OUT_W - INP_W1 - INP_W2; i+=16)
        writeincr_v16(out, null_v16int8());
    } else {
      for (int i = 0; i < OUT_W - INP_W1; i+=16)
        writeincr_v16(out, readincr_v16(in1));  
      for (int i = 0; i < INP_W1 + INP_W2 - OUT_W; i+=16)
        readincr_v16(in1);
    }
  }

  PROFILE_FOOTER2("ConcatInt8Stream<%s,%d,%d,%d,%d>", 
    typeid(TT).name(), H, INP_W1, INP_W2, OUT_W);
}


template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatTwo32bitStreams<TT, LCNT, H, INP_W, OUT_W>::filter(
	input_stream<TT>* in0,
  input_stream<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER2;

  TT a;

#define WRITE_OUT(in_idx, count) \
  for (int w = 0; w < count; w++) { \
    a = get_ss(in_idx); \
    put_ms(0, a); \
  }

  for (int h = 0; h < H; h++) {
    for (int i = 0; i < OUT_W / INP_W; i++) {
      WRITE_OUT((i & 0x1), INP_W);
    }
  }

#undef WRITE_OUT

  PROFILE_FOOTER2("ConcatTwo32bitStreams<%s,%d,%d,%d,%d>", 
    typeid(TT).name(), LCNT, H, INP_W, OUT_W);
}
