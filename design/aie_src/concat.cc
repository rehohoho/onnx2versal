#include "concat.h"
#include "kernel_utils.h"


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
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter8\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter7\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter6\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter5\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter4(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter4\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter3(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter3\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    CAT(in2);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter2(
	input_window<TT>* in0,
	input_window<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter2\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    CAT(in1);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatScalar<TT, LCNT, H, INP_W, OUT_W>::filter1(
	input_window<TT>* in0,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalar<%s,%d,%d,%d,%d>::filter1\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT(in0);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter8\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter7\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter6\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter5\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter4(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter4\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter3(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter3\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    CAT_FLOAT(in2);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter2(
	input_window<TT>* in0,
	input_window<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter2\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_FLOAT(in0);
    CAT_FLOAT(in1);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatFloat<TT, LCNT, H, INP_W, OUT_W>::filter1(
	input_window<TT>* in0,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatFloat<%s,%d,%d,%d,%d>::filter1\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_FLOAT(in0);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter8\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter7\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter6\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
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
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter5\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter4(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
	input_window<TT>* in3,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter4\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter3(
	input_window<TT>* in0,
	input_window<TT>* in1,
	input_window<TT>* in2,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter3\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    CAT_INT8(in2);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter2(
	input_window<TT>* in0,
	input_window<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter2\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_INT8(in0);
    CAT_INT8(in1);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}

template <typename TT, int LCNT, int H, int INP_W, int OUT_W>
void ConcatInt8<TT, LCNT, H, INP_W, OUT_W>::filter1(
	input_window<TT>* in0,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatInt8<%s,%d,%d,%d,%d>::filter1\n", typeid(TT).name(), LCNT, H, INP_W, OUT_W));

  for (int i = 0; i < H; i++) {
    int outi = 0;
    CAT_INT8(in0);
    if (OUT_W > INP_W*LCNT) {
      for (int i = 0; i < OUT_W - LCNT*INP_W; i++)
        writeincr(out, 0);
    }
  }

  PROFILE_FOOTER;
}


template <typename TT, int H, int INP_W1, int INP_W2, int OUT_W>
void ConcatScalarStream<TT, H, INP_W1, INP_W2, OUT_W>::filter(
	input_stream<TT>* in0,
  input_stream<TT>* in1,
  output_stream<TT>* out
) {
  PROFILE_HEADER(printf(
    "Running ConcatScalarStream<%s,%d,%d,%d,%d>::filter\n", 
    typeid(TT).name(), H, INP_W1, INP_W2, OUT_W));

  for (int i = 0; i < H; i++) {
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

  PROFILE_FOOTER;
}