#include "graph_qlinearconv.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkHGraph<SPLIT, QLINEARCONV, CONCAT, HCHUNK, 
                                   TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                   B, C, M, KH, KW, GROUP, 
                                   H0, H1, W0, W1> Graph;
    Graph g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::input_gmio gmio_w;

    QLinearConvChunkHGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHStreamGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkHStreamGraph<SPLIT, QLINEARCONV, CONCAT, HCHUNK, 
                                         TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                         B, C, M, KH, KW, GROUP, 
                                         H0, H1, W0, W1> Graph;
    Graph g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::input_gmio gmio_w;

    QLinearConvChunkHStreamGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHPktStreamGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkHPktStreamGraph<SPLIT, QLINEARCONV, CONCAT, HCHUNK, 
                                            TT, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                            B, C, M, KH, KW, GROUP, 
                                            H0, H1, W0, W1> Graph;
    Graph g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::input_gmio gmio_w;

    QLinearConvChunkHPktStreamGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      int8_t x_zero,
      int8_t w_zero,
      int8_t y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
typedef int8_t TT;
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
const int M = 8;
const int KH = 3;
const int KW = 3;
const int GROUP = 1;
const int PADH = KH/2;
const int PADW = KW/2;
const int W1 = (INP_W+KW-1 +15)/16*16 - INP_W - PADW;

const int OUT_W_STRIDE2_3x3 = (OUT_W - KW)/2+1;
const int OUT_W_STRIDE2_PAD16_3x3 = (OUT_W_STRIDE2_3x3 + 15)/16*16;
const int W1_STRIDE2 = (INP_W+KW-1 +15)/16*16 - INP_W;

const int HCHUNK = 15; // OUT_H' * strides + overlap, overlap = K - strides

std::vector<int8_t> int8weights_3x3_int16int8mac {0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 0, 9, 10, 11, 0, 12, 13, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 18, 19, 20, 0, 21, 22, 23, 0, 24, 25, 26, 0, 0, 0, 0, 0, 27, 28, 29, 0, 30, 31, 32, 0, 33, 34, 35, 0, 0, 0, 0, 0, 36, 37, 38, 0, 39, 40, 41, 0, 42, 43, 44, 0, 0, 0, 0, 0, 45, 46, 47, 0, 48, 49, 50, 0, 51, 52, 53, 0, 0, 0, 0, 0, 54, 55, 56, 0, 57, 58, 59, 0, 60, 61, 62, 0, 0, 0, 0, 0, 63, 64, 65, 0, 66, 67, 68, 0, 69, 70, 71, 0, 0, 0, 0, 0};
std::vector<int32_t> int8bias_3x3 {-900, -2759, -4617, -6475, -8334, -10192, -12050, -13909};


// 3x3 stride 1
/**
 * Pad2DStreamInt8<a,1,26,32,1,1,1,15> start = 876,end = 2140,total = 1264
 * SplitInt8<a,1,1344,720,96>::filter2 start = 921,end = 2223,total = 1302
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 2355,end = 5128,total = 2773
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 2359,end = 5132,total = 2773
 * ConcatInt8Stream<a,8,416,416,832> start = 889,end = 5242,total = 4353
 * Total cycles 4366
 */
QLinearConvChunkHGraphTest<SplitInt8, QLinearConvHx4Stream, ConcatInt8Stream, HCHUNK, 
                           TT, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                           PADH, PADH, PADW, W1> qLinearConvScalarStream(
  "qLinearConvScalarStream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x8x26x28_QLinearConvScalar.txt");

/**
 * Pad2DStreamInt8<a,1,26,32,1,1,1,15> start = 6319,end = 7583,total = 1264
 * SplitFilterInt8StreamTwice<a,1,1344,720,96>::filter0 start = 6322,end = 7604,total = 1282
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 7037,end = 9810,total = 2773
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 7614,end = 10387,total = 2773
 * ConcatInt8Stream<a,8,416,416,832> start = 6322,end = 10398,total = 4076
 * Total cycles 4079
 */
QLinearConvChunkHStreamGraphTest<SplitFilterInt8Stream, QLinearConvHx4Stream, ConcatInt8Stream, HCHUNK, 
                           TT, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                           PADH, PADH, PADW, W1> qLinearConvScalarStreamHStream(
  "qLinearConvScalarStreamHStream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x8x26x28_QLinearConvScalarStreamHStream.txt");

/*
Pad2DStreamInt8<a,1,26,28,32,1,1,1,3> start = 740,end = 1585,total = 845
SplitFilterInt8PktStream<a,1,896,480,64>::filter2 start = 747,end = 1628,total = 881
QLinearConvHx4PktStream<15,32,28,32,1,1,1,1,8,3,3,1> start = 804,end = 3765,total = 2961
QLinearConvHx4PktStream<15,32,28,32,1,1,1,1,8,3,3,1> start = 805,end = 4059,total = 3254
ConcatInt8Stream<a,8,416,416,832> start = 747,end = 4069,total = 3322
Total cycles 4088
*/
QLinearConvChunkHPktStreamGraphTest<SplitFilterInt8PktStream, QLinearConvHx4PktStream, ConcatInt8Stream, HCHUNK, 
                                    TT, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                                    PADH, PADH, PADW, W1> qLinearConvScalarStreamHPktStream(
  "qLinearConvScalarStreamHPktStream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x8x26x28_QLinearConvScalarStreamHPktStream.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // init gmio
  int int8weights_3x3_int16int8mac_size = M*C*16 * sizeof(int8_t);
  int8_t* int8weights_3x3_int16int8mac_buf = (int8_t *) adf::GMIO::malloc(int8weights_3x3_int16int8mac_size);
  memcpy(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac.data(), int8weights_3x3_int16int8mac_size);

  // 3x3 stride 1
  adfCheck(qLinearConvScalarStream.init(), "init qLinearConvScalarStream");
  qLinearConvScalarStream.gmio_w.gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvScalarStream.run(ITER_CNT), "run qLinearConvScalarStream");
	adfCheck(qLinearConvScalarStream.end(), "end qLinearConvScalarStream");

  adfCheck(qLinearConvScalarStreamHStream.init(), "init qLinearConvScalarStreamHStream");
  qLinearConvScalarStreamHStream.gmio_w.gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvScalarStreamHStream.run(ITER_CNT), "run qLinearConvScalarStreamHStream");
	adfCheck(qLinearConvScalarStreamHStream.end(), "end qLinearConvScalarStreamHStream");

  adfCheck(qLinearConvScalarStreamHPktStream.init(), "init qLinearConvScalarStreamHPktStream");
  qLinearConvScalarStreamHPktStream.gmio_w.gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvScalarStreamHPktStream.run(ITER_CNT), "run qLinearConvScalarStreamHPktStream");
	adfCheck(qLinearConvScalarStreamHPktStream.end(), "end qLinearConvScalarStreamHPktStream");

  // cleanup gmio
  adf::GMIO::free(int8weights_3x3_int16int8mac_buf);
  return 0;
}
#endif
