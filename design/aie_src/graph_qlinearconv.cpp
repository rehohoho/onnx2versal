#include "graph_qlinearconv.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int, int, int, int, int> class PAD,
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvGraphTest : public adf::graph {

  private:
    typedef QLinearConvGraph<PAD, QLINEARCONV, 
                             TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                             B, C, M, KH, KW, GROUP, H0, H1, W0, W1> Graph;
    Graph g;
    static constexpr int OUT_H = Graph::OUT_H;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QLinearConvGraphTest(
      const std::string& id,
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearconv_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*C*INP_H*INP_W_PAD>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*M*OUT_H*OUT_W_PAD>> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, int, int, int, int, int, int, int, int> class PAD,
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvStreamGraphTest : public adf::graph {

  private:
    typedef QLinearConvStreamGraph<PAD, QLINEARCONV, 
                                   TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                                   B, C, M, KH, KW, GROUP, H0, H1, W0, W1> Graph;
    Graph g;
    static constexpr int OUT_H = Graph::OUT_H;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    std::vector<adf::input_gmio> gmio_w;

    QLinearConvStreamGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearconv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w.push_back(
        adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000));

      adf::connect<adf::window<B*C*INP_H*INP_W_PAD>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w[0].out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

    QLinearConvStreamGraphTest(
      const std::string& id,
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      TTPARAM w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearconv_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));

      adf::connect<adf::window<B*C*INP_H*INP_W_PAD>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
typedef int8_t TT;
typedef int8_t TTPARAM;
const int INP_H = 26;
const int INP_W = 28;
const int INP_W_PAD = (INP_W + 15)/16*16;
const int OUT_H = INP_H;
const int OUT_W = INP_W;
const int OUT_W_PAD16 = (OUT_W + 15)/16*16;
const int STEP_H = 1;
const int STEP_W = 1;
const int B = 1;
const int C = 3;
const int M = 4;
const int KH = 5;
const int KW = 5;
const int GROUP = 1;
const int PADH = KH/2;
const int PADW = KW/2;
const int W1 = (INP_W + KW-1 +15)/16*16 - INP_W - PADW;

// 5x5 stride 1
std::vector<TTPARAM> int8weights_5x5 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43};
std::vector<int32_t> int8bias_5x5 {-69375, -69034, 129708, -10750};
std::vector<TTPARAM> int8weights_5x5_pad {0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0, 0, 0, 0, 0, 0, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 0, 0, 0, 0, 0, 0, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 0, 0, 0, 0, 0, 0, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 0, 0, 0, 0, 0, 0, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 0, 0, 0, 0, 0, 0, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 0, 0, 0, 0, 0, 0, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 0, 0, 0, 0, 0, 0, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 0, 0, 0, 0, 0, 0, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 0, 0, 0, 0, 0, 0, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 0, 0, 0, 0, 0, 0, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 0, 0, 0, 0, 0, 0, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 0, 0, 0, 0, 0, 0, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 0, 0, 0, 0, 0, 0, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 0, 0, 0, 0, 0, 0, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 0, 0, 0, 0, 0, 0, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 0, 0, 0, 0, 0, 0, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 0, 0, 0, 0, 0, 0, 90, 90, 91, 91, 92, 92, 93, 93, 94, 94, 0, 0, 0, 0, 0, 0, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 0, 0, 0, 0, 0, 0, 100, 100, 101, 101, 102, 102, 103, 103, 104, 104, 0, 0, 0, 0, 0, 0, 105, 105, 106, 106, 107, 107, 108, 108, 109, 109, 0, 0, 0, 0, 0, 0, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 0, 0, 0, 0, 0, 0, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 0, 0, 0, 0, 0, 0, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 0, 0, 0, 0, 0, 0, 125, 125, 126, 126, 127, 127, -128, -128, -127, -127, 0, 0, 0, 0, 0, 0, -126, -126, -125, -125, -124, -124, -123, -123, -122, -122, 0, 0, 0, 0, 0, 0, -121, -121, -120, -120, -119, -119, -118, -118, -117, -117, 0, 0, 0, 0, 0, 0, -116, -116, -115, -115, -114, -114, -113, -113, -112, -112, 0, 0, 0, 0, 0, 0, -111, -111, -110, -110, -109, -109, -108, -108, -107, -107, 0, 0, 0, 0, 0, 0, -106, -106, -105, -105, -104, -104, -103, -103, -102, -102, 0, 0, 0, 0, 0, 0, -101, -101, -100, -100, -99, -99, -98, -98, -97, -97, 0, 0, 0, 0, 0, 0, -96, -96, -95, -95, -94, -94, -93, -93, -92, -92, 0, 0, 0, 0, 0, 0, -91, -91, -90, -90, -89, -89, -88, -88, -87, -87, 0, 0, 0, 0, 0, 0, -86, -86, -85, -85, -84, -84, -83, -83, -82, -82, 0, 0, 0, 0, 0, 0, -81, -81, -80, -80, -79, -79, -78, -78, -77, -77, 0, 0, 0, 0, 0, 0, -76, -76, -75, -75, -74, -74, -73, -73, -72, -72, 0, 0, 0, 0, 0, 0, -71, -71, -70, -70, -69, -69, -68, -68, -67, -67, 0, 0, 0, 0, 0, 0, -66, -66, -65, -65, -64, -64, -63, -63, -62, -62, 0, 0, 0, 0, 0, 0, -61, -61, -60, -60, -59, -59, -58, -58, -57, -57, 0, 0, 0, 0, 0, 0, -56, -56, -55, -55, -54, -54, -53, -53, -52, -52, 0, 0, 0, 0, 0, 0, -51, -51, -50, -50, -49, -49, -48, -48, -47, -47, 0, 0, 0, 0, 0, 0, -46, -46, -45, -45, -44, -44, -43, -43, -42, -42, 0, 0, 0, 0, 0, 0, -41, -41, -40, -40, -39, -39, -38, -38, -37, -37, 0, 0, 0, 0, 0, 0, -36, -36, -35, -35, -34, -34, -33, -33, -32, -32, 0, 0, 0, 0, 0, 0, -31, -31, -30, -30, -29, -29, -28, -28, -27, -27, 0, 0, 0, 0, 0, 0, -26, -26, -25, -25, -24, -24, -23, -23, -22, -22, 0, 0, 0, 0, 0, 0, -21, -21, -20, -20, -19, -19, -18, -18, -17, -17, 0, 0, 0, 0, 0, 0, -16, -16, -15, -15, -14, -14, -13, -13, -12, -12, 0, 0, 0, 0, 0, 0, -11, -11, -10, -10, -9, -9, -8, -8, -7, -7, 0, 0, 0, 0, 0, 0, -6, -6, -5, -5, -4, -4, -3, -3, -2, -2, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 0, 0, 0, 0, 0, 0, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 0, 0, 0, 0, 0, 0, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 0, 0, 0, 0, 0, 0, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 0, 0, 0, 0, 0, 0, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 0, 0, 0, 0, 0, 0, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 0, 0, 0, 0, 0, 0, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 0, 0};

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP,
                     PADH, PADH, PADW, W1> qLinearConvScalar(
  "qLinearConvScalar", int8weights_5x5, int8bias_5x5, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_shape1x4x26x28_QLinearConvScalar.txt");

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConv5x5, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP,
                     PADH, PADH, PADW, W1> qLinearConv5x5(
  "qLinearConv5x5", int8weights_5x5_pad, int8bias_5x5, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_shape1x4x26x28_QLinearConv5x5.txt");

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConv5x5Scale32bit, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP,
                     PADH, PADH, PADW, W1> qLinearConv5x5Scale32bit(
  "qLinearConv5x5Scale32bit", int8weights_5x5_pad, int8bias_5x5, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_shape1x4x26x28_QLinearConv5x5Scale32bit.txt");

// 3x3 stride 1
const int KH3x3 = 3;
const int KW3x3 = 3;
const int PADH3x3 = KH3x3/2;
const int PADW3x3 = KW3x3/2;
const int W1_3x3 = (INP_W + KW3x3-1 +15)/16*16 - INP_W - PADW3x3;
const int OUT_W_STRIDE2_3x3 = (OUT_W - KW3x3)/2+1;
const int OUT_W_STRIDE2_PAD16_3x3 = (OUT_W_STRIDE2_3x3 + 15)/16*16;
const int W1_STRIDE2_3x3 = (INP_W + KW3x3-1 +15)/16*16 - INP_W;

std::vector<TTPARAM> int8weights_3x3 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107};
std::vector<int32_t> int8bias_3x3 {-8775, -26834, -44892, -62950};
std::vector<TTPARAM> int8weights_3x3_pad {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0, 0, 0, 0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 0, 0, 0, 0, 0, 0, 45, 46, 47, 48, 49, 50, 51, 52, 53, 0, 0, 0, 0, 0, 0, 0, 54, 55, 56, 57, 58, 59, 60, 61, 62, 0, 0, 0, 0, 0, 0, 0, 63, 64, 65, 66, 67, 68, 69, 70, 71, 0, 0, 0, 0, 0, 0, 0, 72, 73, 74, 75, 76, 77, 78, 79, 80, 0, 0, 0, 0, 0, 0, 0, 81, 82, 83, 84, 85, 86, 87, 88, 89, 0, 0, 0, 0, 0, 0, 0, 90, 91, 92, 93, 94, 95, 96, 97, 98, 0, 0, 0, 0, 0, 0, 0, 99, 100, 101, 102, 103, 104, 105, 106, 107, 0, 0, 0, 0, 0, 0, 0};
std::vector<TTPARAM> int8weights_3x3_int16int8mac {0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 0, 9, 10, 11, 0, 12, 13, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 18, 19, 20, 0, 21, 22, 23, 0, 24, 25, 26, 0, 0, 0, 0, 0, 27, 28, 29, 0, 30, 31, 32, 0, 33, 34, 35, 0, 0, 0, 0, 0, 36, 37, 38, 0, 39, 40, 41, 0, 42, 43, 44, 0, 0, 0, 0, 0, 45, 46, 47, 0, 48, 49, 50, 0, 51, 52, 53, 0, 0, 0, 0, 0, 54, 55, 56, 0, 57, 58, 59, 0, 60, 61, 62, 0, 0, 0, 0, 0, 63, 64, 65, 0, 66, 67, 68, 0, 69, 70, 71, 0, 0, 0, 0, 0, 72, 73, 74, 0, 75, 76, 77, 0, 78, 79, 80, 0, 0, 0, 0, 0, 81, 82, 83, 0, 84, 85, 86, 0, 87, 88, 89, 0, 0, 0, 0, 0, 90, 91, 92, 0, 93, 94, 95, 0, 96, 97, 98, 0, 0, 0, 0, 0, 99, 100, 101, 0, 102, 103, 104, 0, 105, 106, 107, 0, 0, 0, 0, 0};
std::vector<TTPARAM> int8weights_3x3_int8int8mac {0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 7, 7, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 10, 10, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 13, 13, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 16, 16, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 19, 19, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 22, 22, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 24, 25, 25, 26, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 27, 28, 28, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 31, 31, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 33, 34, 34, 35, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 36, 37, 37, 38, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 39, 40, 40, 41, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 42, 43, 43, 44, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 45, 46, 46, 47, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 48, 49, 49, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 51, 52, 52, 53, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 54, 55, 55, 56, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 57, 58, 58, 59, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 60, 61, 61, 62, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 63, 64, 64, 65, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 66, 67, 67, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 69, 70, 70, 71, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 72, 73, 73, 74, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 75, 76, 76, 77, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 78, 79, 79, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 81, 82, 82, 83, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 84, 85, 85, 86, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 87, 88, 88, 89, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 90, 91, 91, 92, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 93, 94, 94, 95, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 96, 97, 97, 98, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 99, 100, 100, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 102, 103, 103, 104, 104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 105, 106, 106, 107, 107, 0, 0, 0, 0, 0, 0};

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH3x3, KW3x3, GROUP,
                     PADH3x3, PADH3x3, PADW3x3, W1_3x3> qLinearConvScalar_3x3(
  "qLinearConvScalar_3x3", int8weights_3x3, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvScalar.txt");

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConv3x3, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH3x3, KW3x3, GROUP,
                     PADH3x3, PADH3x3, PADW3x3, W1_3x3> qLinearConv3x3(
  "qLinearConv3x3", int8weights_3x3_int16int8mac, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConv3x3.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvScalarStream, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH3x3, KW3x3, GROUP,
                           PADH3x3, PADH3x3, PADW3x3, W1_3x3> qLinearConvScalarStream(
  "qLinearConvScalarStream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvScalarStream.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvHx4Stream, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH3x3, KW3x3, GROUP,
                           PADH3x3, PADH3x3, PADW3x3, W1_3x3> qLinearConvHx4Stream(
  "qLinearConvHx4Stream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx4Stream.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvHx4StreamScale32bit, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH3x3, KW3x3, GROUP,
                           PADH3x3, PADH3x3, PADW3x3, W1_3x3> qLinearConvHx4StreamScale32bit(
  "qLinearConvHx4StreamScale32bit", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx4StreamScale32bit.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvHx6x8bitStream, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH3x3, KW3x3, GROUP,
                           PADH3x3, PADH3x3, PADW3x3, W1_3x3> qLinearConvHx6x8bitStream(
  "qLinearConvHx6x8bitStream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx6x8bitStream.txt");

// 3x3 stride 2
QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W_STRIDE2_3x3, OUT_W_STRIDE2_PAD16_3x3, 2, 2, B, C, M, KH3x3, KW3x3, GROUP,
                     0, 0, 0, W1_STRIDE2_3x3> qLinearConvScalar_3x3_s2(
  "qLinearConvScalar_3x3_s2", int8weights_3x3, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_stride2_shape1x4x12x13_QLinearConvScalar.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvHx4Stream, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W_STRIDE2_3x3, OUT_W_STRIDE2_PAD16_3x3, 2, 2, B, C, M, KH3x3, KW3x3, GROUP,
                           0, 0, 0, W1_STRIDE2_3x3> qLinearConvHx4Stream_s2(
  "qLinearConvHx4Stream_s2", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_stride2_shape1x4x12x13_QLinearConvHx4Stream.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvHx4StreamScale32bit, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W_STRIDE2_3x3, OUT_W_STRIDE2_PAD16_3x3, 2, 2, B, C, M, KH3x3, KW3x3, GROUP,
                           0, 0, 0, W1_STRIDE2_3x3> qLinearConvHx4StreamScale32bit_s2(
  "qLinearConvHx4StreamScale32bit_s2", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_stride2_shape1x4x12x13_QLinearConvHx4StreamScale32bit.txt");


// 1x1 stride 1
std::vector<TTPARAM> int8weights_1x1 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
std::vector<TTPARAM> int8weights_1x1_pad {0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
std::vector<int32_t> int8bias_1x1 {-75, -134, -192, -250};

const int KH1x1 = 1;
const int KW1x1 = 1;
const int OUT_W_STRIDE2_1x1 = (OUT_W - KW1x1)/2+1;
const int OUT_W_STRIDE2_PAD16_1x1 = (OUT_W_STRIDE2_1x1 + 15)/16*16;
const int W1_1x1 = (INP_W + 15)/16*16 - INP_W;

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH1x1, KW1x1, GROUP,
                     0, 0, 0, W1_1x1> qLinearConvScalar_1x1(
  "qLinearConvScalar_1x1", int8weights_1x1, int8bias_1x1, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_1x1_shape1x4x26x28_QLinearConvScalar.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConv1x1Stream, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH1x1, KW1x1, GROUP,
                           0, 0, 0, W1_1x1> qLinearConv1x1Stream(
  "qLinearConv1x1Stream", int8bias_1x1, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_1x1_shape1x4x26x28_QLinearConv1x1Stream.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConv1x1Stream, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_STRIDE2_PAD16_1x1, 2, 2, B, C, M, KH1x1, KW1x1, GROUP,
                           0, 0, 0, W1_1x1> qLinearConv1x1Stream_s2(
  "qLinearConv1x1Stream_s2", int8bias_1x1, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_1x1_stride2_shape1x4x13x14_QLinearConv1x1Stream.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // init gmio
  int int8weights_3x3_pad_size = M*C*16 * sizeof(TT);
  TT* int8weights_3x3_pad_buf = (TT *) adf::GMIO::malloc(int8weights_3x3_pad_size);
  memcpy(int8weights_3x3_pad_buf, int8weights_3x3_pad.data(), int8weights_3x3_pad_size);
  
  int int8weights_3x3_int16int8mac_size = M*C*16 * sizeof(TT);
  TT* int8weights_3x3_int16int8mac_buf = (TT *) adf::GMIO::malloc(int8weights_3x3_int16int8mac_size);
  memcpy(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac.data(), int8weights_3x3_int16int8mac_size);

  int int8weights_3x3_int8int8mac_size = M*C*KH3x3*16 * sizeof(TT);
  TT* int8weights_3x3_int8int8mac_buf = (TT *) adf::GMIO::malloc(int8weights_3x3_int8int8mac_size);
  memcpy(int8weights_3x3_int8int8mac_buf, int8weights_3x3_int8int8mac.data(), int8weights_3x3_int8int8mac_size);

  int int8weights_1x1_pad_size = M*((C+15)/16*16) * sizeof(TT);
  TT* int8weights_1x1_pad_buf = (TT *) adf::GMIO::malloc(int8weights_1x1_pad_size);
  memcpy(int8weights_1x1_pad_buf, int8weights_1x1_pad.data(), int8weights_1x1_pad_size);

  // 5x5 stride 1
  adfCheck(qLinearConvScalar.init(), "init qLinearConvScalar");
  adfCheck(qLinearConvScalar.run(ITER_CNT), "run qLinearConvScalar");
	adfCheck(qLinearConvScalar.end(), "end qLinearConvScalar");

  adfCheck(qLinearConv5x5.init(), "init qLinearConv5x5");
  adfCheck(qLinearConv5x5.run(ITER_CNT), "run qLinearConv5x5");
	adfCheck(qLinearConv5x5.end(), "end qLinearConv5x5");

  adfCheck(qLinearConv5x5Scale32bit.init(), "init qLinearConv5x5Scale32bit");
  adfCheck(qLinearConv5x5Scale32bit.run(ITER_CNT), "run qLinearConv5x5Scale32bit");
	adfCheck(qLinearConv5x5Scale32bit.end(), "end qLinearConv5x5Scale32bit");

  // 3x3 stride 1
  adfCheck(qLinearConvScalar_3x3.init(), "init qLinearConvScalar_3x3");
  adfCheck(qLinearConvScalar_3x3.run(ITER_CNT), "run qLinearConvScalar_3x3");
	adfCheck(qLinearConvScalar_3x3.end(), "end qLinearConvScalar_3x3");

  adfCheck(qLinearConv3x3.init(), "init qLinearConv3x3");
  adfCheck(qLinearConv3x3.run(ITER_CNT), "run qLinearConv3x3");
	adfCheck(qLinearConv3x3.end(), "end qLinearConv3x3");

  adfCheck(qLinearConvScalarStream.init(), "init qLinearConvScalarStream");
  qLinearConvScalarStream.gmio_w[0].gm2aie_nb(int8weights_3x3_pad_buf, int8weights_3x3_pad_size);
  adfCheck(qLinearConvScalarStream.run(ITER_CNT), "run qLinearConvScalarStream");
	adfCheck(qLinearConvScalarStream.end(), "end qLinearConvScalarStream");

  adfCheck(qLinearConvHx4Stream.init(), "init qLinearConvHx4Stream");
  qLinearConvHx4Stream.gmio_w[0].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4Stream.run(ITER_CNT), "run qLinearConvHx4Stream");
	adfCheck(qLinearConvHx4Stream.end(), "end qLinearConvHx4Stream");

  adfCheck(qLinearConvHx4StreamScale32bit.init(), "init qLinearConvHx4StreamScale32bit");
  qLinearConvHx4StreamScale32bit.gmio_w[0].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4StreamScale32bit.run(ITER_CNT), "run qLinearConvHx4StreamScale32bit");
	adfCheck(qLinearConvHx4StreamScale32bit.end(), "end qLinearConvHx4StreamScale32bit");

  adfCheck(qLinearConvHx6x8bitStream.init(), "init qLinearConvHx6x8bitStream");
  qLinearConvHx6x8bitStream.gmio_w[0].gm2aie_nb(int8weights_3x3_int8int8mac_buf, int8weights_3x3_int8int8mac_size);
  adfCheck(qLinearConvHx6x8bitStream.run(ITER_CNT), "run qLinearConvHx6x8bitStream");
	adfCheck(qLinearConvHx6x8bitStream.end(), "end qLinearConvHx6x8bitStream");

  // 3x3 stride 2
  adfCheck(qLinearConvScalar_3x3_s2.init(), "init qLinearConvScalar_3x3_s2");
  adfCheck(qLinearConvScalar_3x3_s2.run(ITER_CNT), "run qLinearConvScalar_3x3_s2");
	adfCheck(qLinearConvScalar_3x3_s2.end(), "end qLinearConvScalar_3x3_s2");

  adfCheck(qLinearConvHx4Stream_s2.init(), "init qLinearConvHx4Stream_s2");
  qLinearConvHx4Stream_s2.gmio_w[0].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4Stream_s2.run(ITER_CNT), "run qLinearConvHx4Stream_s2");
	adfCheck(qLinearConvHx4Stream_s2.end(), "end qLinearConvHx4Stream_s2");

  adfCheck(qLinearConvHx4StreamScale32bit_s2.init(), "init qLinearConvHx4StreamScale32bit_s2");
  qLinearConvHx4StreamScale32bit_s2.gmio_w[0].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4StreamScale32bit_s2.run(ITER_CNT), "run qLinearConvHx4StreamScale32bit_s2");
	adfCheck(qLinearConvHx4StreamScale32bit_s2.end(), "end qLinearConvHx4StreamScale32bit_s2");

  // 1x1 stride 1
  adfCheck(qLinearConvScalar_1x1.init(), "init qLinearConvScalar_1x1");
  adfCheck(qLinearConvScalar_1x1.run(ITER_CNT), "run qLinearConvScalar_1x1");
	adfCheck(qLinearConvScalar_1x1.end(), "end qLinearConvScalar_1x1");

  adfCheck(qLinearConv1x1Stream.init(), "init qLinearConv1x1Stream");
  qLinearConv1x1Stream.gmio_w[0].gm2aie_nb(int8weights_1x1_pad_buf, int8weights_1x1_pad_size);
  adfCheck(qLinearConv1x1Stream.run(ITER_CNT), "run qLinearConv1x1Stream");
	adfCheck(qLinearConv1x1Stream.end(), "end qLinearConv1x1Stream");

  adfCheck(qLinearConv1x1Stream_s2.init(), "init qLinearConv1x1Stream_s2");
  qLinearConv1x1Stream_s2.gmio_w[0].gm2aie_nb(int8weights_1x1_pad_buf, int8weights_1x1_pad_size);
  adfCheck(qLinearConv1x1Stream_s2.run(ITER_CNT), "run qLinearConv1x1Stream_s2");
	adfCheck(qLinearConv1x1Stream_s2.end(), "end qLinearConv1x1Stream_s2");

  // cleanup gmio
  adf::GMIO::free(int8weights_3x3_pad_buf);
  adf::GMIO::free(int8weights_3x3_int16int8mac_buf);
  adf::GMIO::free(int8weights_1x1_pad_buf);
  return 0;
}
#endif
