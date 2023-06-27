#include "graph_qlinearconv.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int, int, int, int> class PAD,
  template<int, int, int, int, int, int, int, int, int,int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int K,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvGraphTest : public adf::graph {

  private:
    typedef QLinearConvGraph<PAD, QLINEARCONV, 
                             INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                             B, C, M, K, H0, H1, W0, W1> Graph;
    Graph g;
    static constexpr int OUT_H = Graph::OUT_H;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    QLinearConvGraphTest(
      const std::string& id,
      std::vector<int8_t> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearconv_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*C*INP_H*INP_W>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*M*OUT_H*OUT_W_PAD>> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, int, int, int, int, int, int, int> class PAD,
  template<int, int, int, int, int, int, int, int, int,int> class QLINEARCONV, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int K,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvStreamGraphTest : public adf::graph {

  private:
    typedef QLinearConvStreamGraph<PAD, QLINEARCONV, 
                                   INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                                   B, C, M, K, H0, H1, W0, W1> Graph;
    Graph g;
    static constexpr int OUT_H = Graph::OUT_H;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::input_gmio gmio_w;

    QLinearConvStreamGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "qlinearconv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000);

      adf::connect<adf::window<B*C*INP_H*INP_W>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream>                  (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::window<B*M*OUT_H*OUT_W_PAD>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
const int INP_H = 26;
const int INP_W = 28;
const int INP_W_PAD16 = (INP_W + 15)/16*16;
const int OUT_H = INP_H;
const int OUT_W = INP_W;
const int OUT_W_PAD16 = (OUT_W + 15)/16*16;
const int STEP_H = 1;
const int STEP_W = 1;
const int B = 1;
const int C = 1;
const int M = 6;
const int K = 5;
const int PAD = K/2;
const int W1 = (INP_W_PAD16 + K-1 +15)/16*16 - INP_W_PAD16 - PAD;

// 5x5 stride 1
std::vector<int8_t> int8weights_5x5 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107};
std::vector<int32_t> int8bias_5x5 {-7500, -22959, -38417, -53875, -69334, 56008};
std::vector<int8_t> int8weights_5x5_pad {0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0, 0, 0, 0, 0, 0, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 0, 0, 0, 0, 0, 0, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 0, 0, 0, 0, 0, 0, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 0, 0, 0, 0, 0, 0, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 0, 0, 0, 0, 0, 0, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 0, 0, 0, 0, 0, 0, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 0, 0, 0, 0, 0, 0, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 0, 0, 0, 0, 0, 0, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 0, 0, 0, 0, 0, 0, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 0, 0, 0, 0, 0, 0, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 0, 0, 0, 0, 0, 0, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 0, 0, 0, 0, 0, 0, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 0, 0, 0, 0, 0, 0, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 0, 0, 0, 0, 0, 0, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 0, 0, 0, 0, 0, 0, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 0, 0, 0, 0, 0, 0, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 0, 0, 0, 0, 0, 0, 90, 90, 91, 91, 92, 92, 93, 93, 94, 94, 0, 0, 0, 0, 0, 0, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 0, 0, 0, 0, 0, 0, 100, 100, 101, 101, 102, 102, 103, 103, 104, 104, 0, 0, 0, 0, 0, 0, 105, 105, 106, 106, 107, 107, 108, 108, 109, 109, 0, 0, 0, 0, 0, 0, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 0, 0, 0, 0, 0, 0, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 0, 0, 0, 0, 0, 0, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 0, 0, 0, 0, 0, 0, 125, 125, 126, 126, 127, 127, -128, -128, -127, -127, 0, 0, 0, 0, 0, 0, -126, -126, -125, -125, -124, -124, -123, -123, -122, -122, 0, 0, 0, 0, 0, 0, -121, -121, -120, -120, -119, -119, -118, -118, -117, -117, 0, 0, 0, 0, 0, 0, -116, -116, -115, -115, -114, -114, -113, -113, -112, -112, 0, 0, 0, 0, 0, 0, -111, -111, -110, -110, -109, -109, -108, -108, -107, -107, 0, 0};

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K,
                     PAD, PAD, PAD, W1> qLinearConvScalar(
  "qLinearConvScalar", int8weights_5x5, int8bias_5x5, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_shape1x6x26x28_QLinearConvScalar.txt");

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConv5x5, 
                     INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K,
                     PAD, PAD, PAD, W1> qLinearConvVector(
  "qLinearConvVector", int8weights_5x5_pad, int8bias_5x5, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_shape1x6x26x28_QLinearConv5x5.txt");

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConv5x5Scale32bit, 
                     INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K,
                     PAD, PAD, PAD, W1> qLinearConvVectorScale32bit(
  "qLinearConvVectorScale32bit", int8weights_5x5_pad, int8bias_5x5, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_shape1x6x26x28_QLinearConv5x5Scale32bit.txt");

// 3x3 stride 1
const int K_3x3 = 3;
const int PAD_3x3 = K_3x3/2;
const int W1_3x3 = (INP_W_PAD16+ K_3x3-1 +15)/16*16 - INP_W_PAD16 - PAD_3x3;
const int OUT_W_STRIDE2_3x3 = (OUT_W - K_3x3)/2+1;
const int OUT_W_STRIDE2_PAD16_3x3 = (OUT_W_STRIDE2_3x3 + 15)/16*16;

std::vector<int8_t> int8weights_3x3 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53};
std::vector<int32_t> int8bias_3x3 {-900, -2759, -4617, -6475, -8334, -10192};
std::vector<int8_t> int8weights_3x3_pad {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0, 0, 0, 0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 0, 0, 0, 0, 0, 0, 45, 46, 47, 48, 49, 50, 51, 52, 53, 0, 0, 0, 0, 0, 0, 0};
std::vector<int8_t> int8weights_3x3_int16int8mac {0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 0, 9, 10, 11, 0, 12, 13, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 18, 19, 20, 0, 21, 22, 23, 0, 24, 25, 26, 0, 0, 0, 0, 0, 27, 28, 29, 0, 30, 31, 32, 0, 33, 34, 35, 0, 0, 0, 0, 0, 36, 37, 38, 0, 39, 40, 41, 0, 42, 43, 44, 0, 0, 0, 0, 0, 45, 46, 47, 0, 48, 49, 50, 0, 51, 52, 53, 0, 0, 0, 0, 0};

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K_3x3,
                     PAD_3x3, PAD_3x3, PAD_3x3, W1_3x3> qLinearConvScalar_3x3(
  "qLinearConvScalar_3x3", int8weights_3x3, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x6x26x28_QLinearConvScalar.txt");

QLinearConvGraphTest<Pad2DStreamInt8, QLinearConv3x3, 
                     INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K_3x3,
                     PAD_3x3, PAD_3x3, PAD_3x3, W1_3x3> qLinearConv3x3(
  "qLinearConv3x3", int8weights_3x3_int16int8mac, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x6x26x28_QLinearConv3x3.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConvScalarStream, 
                           INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K_3x3,
                           PAD_3x3, PAD_3x3, PAD_3x3, W1_3x3> qLinearConvScalarStream_3x3(
  "qLinearConvScalarStream_3x3", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x6x26x28_QLinearConvScalarStream.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConv3x3Stream, 
                           INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K_3x3,
                           PAD_3x3, PAD_3x3, PAD_3x3, W1_3x3> qLinearConv3x3Stream(
  "qLinearConv3x3Stream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x6x26x28_QLinearConv3x3Stream.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConv3x3StreamMultiRow, 
                           INP_H, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, K_3x3,
                           PAD_3x3, PAD_3x3, PAD_3x3, W1_3x3> qLinearConv3x3StreamMultiRow(
  "qLinearConv3x3StreamMultiRow", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x6x26x28_QLinearConv3x3StreamMultiRow.txt");

// 3x3 stride 2
QLinearConvGraphTest<Pad2DStreamInt8, QLinearConvScalar, 
                     INP_H, INP_W_PAD16, OUT_W_STRIDE2_3x3, OUT_W_STRIDE2_PAD16_3x3, 2, 2, B, C, M, K_3x3,
                     0, 0, 0, 0> qLinearConvScalar_3x3_s2(
  "qLinearConvScalar_3x3_s2", int8weights_3x3, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_stride2_shape1x6x12x13_QLinearConvScalar.txt");

QLinearConvStreamGraphTest<Pad2DStreamInt8, QLinearConv3x3Stream, 
                           INP_H, INP_W_PAD16, OUT_W_STRIDE2_3x3, OUT_W_STRIDE2_PAD16_3x3, 2, 2, B, C, M, K_3x3,
                           0, 0, 0, 0> qLinearConv3x3Stream_s2(
  "qLinearConv3x3Stream_s2", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_stride2_shape1x6x12x13_QLinearConv3x3Stream.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // init gmio
  int int8weights_3x3_pad_size = M*C*16 * sizeof(float_t);
  float_t* int8weights_3x3_pad_buf = (float_t *) adf::GMIO::malloc(int8weights_3x3_pad_size);
  memcpy(int8weights_3x3_pad_buf, int8weights_3x3_pad.data(), int8weights_3x3_pad_size);
  int int8weights_3x3_int16int8mac_size = M*C*16 * sizeof(float_t);
  float_t* int8weights_3x3_int16int8mac_buf = (float_t *) adf::GMIO::malloc(int8weights_3x3_int16int8mac_size);
  memcpy(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac.data(), int8weights_3x3_int16int8mac_size);

  // 5x5 stride 1
  adfCheck(qLinearConvScalar.init(), "init qLinearConvScalar");
  adfCheck(qLinearConvScalar.run(ITER_CNT), "run qLinearConvScalar");
	adfCheck(qLinearConvScalar.end(), "end qLinearConvScalar");

  adfCheck(qLinearConvVector.init(), "init qLinearConvVector");
  adfCheck(qLinearConvVector.run(ITER_CNT), "run qLinearConvVector");
	adfCheck(qLinearConvVector.end(), "end qLinearConvVector");

  adfCheck(qLinearConvVectorScale32bit.init(), "init qLinearConvVectorScale32bit");
  adfCheck(qLinearConvVectorScale32bit.run(ITER_CNT), "run qLinearConvVectorScale32bit");
	adfCheck(qLinearConvVectorScale32bit.end(), "end qLinearConvVectorScale32bit");

  // 3x3 stride 1
  adfCheck(qLinearConvScalar_3x3.init(), "init qLinearConvScalar_3x3");
  adfCheck(qLinearConvScalar_3x3.run(ITER_CNT), "run qLinearConvScalar_3x3");
	adfCheck(qLinearConvScalar_3x3.end(), "end qLinearConvScalar_3x3");

  adfCheck(qLinearConv3x3.init(), "init qLinearConv3x3");
  adfCheck(qLinearConv3x3.run(ITER_CNT), "run qLinearConv3x3");
	adfCheck(qLinearConv3x3.end(), "end qLinearConv3x3");

  adfCheck(qLinearConvScalarStream_3x3.init(), "init qLinearConvScalarStream_3x3");
  qLinearConvScalarStream_3x3.gmio_w.gm2aie_nb(int8weights_3x3_pad_buf, int8weights_3x3_pad_size);
  adfCheck(qLinearConvScalarStream_3x3.run(ITER_CNT), "run qLinearConvScalarStream_3x3");
	adfCheck(qLinearConvScalarStream_3x3.end(), "end qLinearConvScalarStream_3x3");

  adfCheck(qLinearConv3x3Stream.init(), "init qLinearConv3x3Stream");
  qLinearConv3x3Stream.gmio_w.gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConv3x3Stream.run(ITER_CNT), "run qLinearConv3x3Stream");
	adfCheck(qLinearConv3x3Stream.end(), "end qLinearConv3x3Stream");

  adfCheck(qLinearConv3x3StreamMultiRow.init(), "init qLinearConv3x3StreamMultiRow");
  qLinearConv3x3StreamMultiRow.gmio_w.gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConv3x3StreamMultiRow.run(ITER_CNT), "run qLinearConv3x3StreamMultiRow");
	adfCheck(qLinearConv3x3StreamMultiRow.end(), "end qLinearConv3x3StreamMultiRow");

  // 3x3 stride 2
  adfCheck(qLinearConvScalar_3x3_s2.init(), "init qLinearConvScalar_3x3_s2");
  adfCheck(qLinearConvScalar_3x3_s2.run(ITER_CNT), "run qLinearConvScalar_3x3_s2");
	adfCheck(qLinearConvScalar_3x3_s2.end(), "end qLinearConvScalar_3x3_s2");

  adfCheck(qLinearConv3x3Stream_s2.init(), "init qLinearConv3x3Stream_s2");
  qLinearConv3x3Stream_s2.gmio_w.gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConv3x3Stream_s2.run(ITER_CNT), "run qLinearConv3x3Stream_s2");
	adfCheck(qLinearConv3x3Stream_s2.end(), "end qLinearConv3x3Stream_s2");

  // cleanup gmio
  adf::GMIO::free(int8weights_3x3_pad_buf);
  adf::GMIO::free(int8weights_3x3_int16int8mac_buf);
  return 0;
}
#endif
