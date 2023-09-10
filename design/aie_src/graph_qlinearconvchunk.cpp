#include "graph_qlinearconv.h"
#include "graph_utils.h"


template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkHGraph<QLINEARCONV, CONCAT, HCHUNK, 
                                   TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
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
      TT x_zero,
      int8_t w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_qlinearconv"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHStreamGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkHStreamGraph<QLINEARCONV, CONCAT, HCHUNK, 
                                         TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
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
      TT x_zero,
      int8_t w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_qlinearconv"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkHPktStreamGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkHPktStreamGraph<QLINEARCONV, CONCAT, HCHUNK, 
                                            TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                            B, C, M, KH, KW, GROUP, 
                                            H0, H1, W0, W1> Graph;
    Graph g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::vector<adf::input_gmio> gmio_w;

    QLinearConvChunkHPktStreamGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      int8_t w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w.push_back(
        adf::input_gmio::create("gmio0_qlinearconv"+id+"_w", 64, 1000));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w[0].out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

    QLinearConvChunkHPktStreamGraphTest(
      const std::string& id,
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      int8_t w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV0, 
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV1, 
  template<typename, typename, int, int, int, int, int, int, int, int, int, int, int, int> class QLINEARCONV2, 
  template<typename, int, int, int, int> class CONCAT, 
  int CCHUNK,
  typename TT, typename TTPARAM, int INP_H, int INP_W, int INP_W_PAD, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class QLinearConvChunkCGraphTest : public adf::graph {

  private:
    typedef QLinearConvChunkCGraph<QLINEARCONV0, QLINEARCONV1, QLINEARCONV2, CONCAT, CCHUNK, 
                                   TT, TTPARAM, INP_H, INP_W, INP_W_PAD, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                   B, C, M, KH, KW, GROUP, 
                                   H0, H1, W0, W1> Graph;
    Graph g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::vector<adf::input_gmio> gmio_w;

    QLinearConvChunkCGraphTest(
      const std::string& id,
      std::vector<TTPARAM> weights,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      int8_t w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(weights, bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

    QLinearConvChunkCGraphTest(
      const std::string& id,
      std::vector<int32_t> bias,
      float x_scale,
      float w_scale,
      float y_scale,
      TT x_zero,
      int8_t w_zero,
      TT y_zero,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias, x_scale, w_scale, y_scale, x_zero, w_zero, y_zero) { 
      plin[0] = adf::input_plio::create("plin0_qlinearconv_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_qlinearconv_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);

      for (int i = 0; i < Graph::LCNT; i++) {
        gmio_w.push_back(
          adf::input_gmio::create("gmio"+std::to_string(i)+"_qlinearconv"+id+"_w", 64, 500));
        adf::connect<adf::stream> (gmio_w[i].out[0], g.pin[1+i]);
      }
    }
};


// instance to be compiled and used in host within xclbin
typedef int8_t TT;
typedef int8_t TTPARAM;
const int INP_H = 26;
const int INP_W = 28;
const int INP_W_PAD16 = (INP_W + 15)/16*16;
const int OUT_H = INP_H;
const int OUT_W = INP_W;
const int OUT_W_PAD16 = (OUT_W + 15)/16*16;
const int STEP_H = 1;
const int STEP_W = 1;
const int B = 1;
const int C = 3;
const int M = 4;
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

std::vector<TTPARAM> int8weights_3x3_int16int8mac {0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 0, 9, 10, 11, 0, 12, 13, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 18, 19, 20, 0, 21, 22, 23, 0, 24, 25, 26, 0, 0, 0, 0, 0, 27, 28, 29, 0, 30, 31, 32, 0, 33, 34, 35, 0, 0, 0, 0, 0, 36, 37, 38, 0, 39, 40, 41, 0, 42, 43, 44, 0, 0, 0, 0, 0, 45, 46, 47, 0, 48, 49, 50, 0, 51, 52, 53, 0, 0, 0, 0, 0, 54, 55, 56, 0, 57, 58, 59, 0, 60, 61, 62, 0, 0, 0, 0, 0, 63, 64, 65, 0, 66, 67, 68, 0, 69, 70, 71, 0, 0, 0, 0, 0, 72, 73, 74, 0, 75, 76, 77, 0, 78, 79, 80, 0, 0, 0, 0, 0, 81, 82, 83, 0, 84, 85, 86, 0, 87, 88, 89, 0, 0, 0, 0, 0, 90, 91, 92, 0, 93, 94, 95, 0, 96, 97, 98, 0, 0, 0, 0, 0, 99, 100, 101, 0, 102, 103, 104, 0, 105, 106, 107, 0, 0, 0, 0, 0};
std::vector<int32_t> int8bias_3x3 {-8775, -26834, -44892, -62950};


// 3x3 stride 1
/**
 * Pad2DStreamInt8<a,1,26,32,1,1,1,15> start = 876,end = 2140,total = 1264
 * SplitInt8<a,1,1344,720,96>::filter2 start = 921,end = 2223,total = 1302
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 2355,end = 5128,total = 2773
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 2359,end = 5132,total = 2773
 * ConcatInt8Stream<a,8,416,416,832> start = 889,end = 5242,total = 4353
 * Total cycles 4366
 */
QLinearConvChunkHGraphTest<QLinearConvHx4Stream, ConcatInt8Stream, HCHUNK, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                           PADH, PADH, PADW, W1> qLinearConvHx4Stream(
  "qLinearConvHx4Stream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx4Stream.txt");

/**
 * Pad2DStreamInt8<a,1,26,32,1,1,1,15> start = 6319,end = 7583,total = 1264
 * SplitFilterInt8StreamTwice<a,1,1344,720,96>::filter0 start = 6322,end = 7604,total = 1282
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 7037,end = 9810,total = 2773
 * QLinearConvHx4Stream<15,48,28,32,1,1,1,1,8,3,3,1> start = 7614,end = 10387,total = 2773
 * ConcatInt8Stream<a,8,416,416,832> start = 6322,end = 10398,total = 4076
 * Total cycles 4079
 */
QLinearConvChunkHStreamGraphTest<QLinearConvHx4Stream, ConcatInt8Stream, HCHUNK, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                           PADH, PADH, PADW, W1> qLinearConvHx4StreamChunkChunkH(
  "qLinearConvHx4StreamChunkChunkH", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx4StreamChunkH.txt");

/*
Pad2DStreamInt8<a,1,26,28,32,1,1,1,3> start = 740,end = 1585,total = 845
SplitFilterInt8PktStream<a,1,896,480,64>::filter2 start = 747,end = 1628,total = 881
QLinearConvHx4PktStream<15,32,28,32,1,1,1,1,8,3,3,1> start = 804,end = 3765,total = 2961
QLinearConvHx4PktStream<15,32,28,32,1,1,1,1,8,3,3,1> start = 805,end = 4059,total = 3254
ConcatInt8Stream<a,8,416,416,832> start = 747,end = 4069,total = 3322
Total cycles 4088
*/
QLinearConvChunkHPktStreamGraphTest<QLinearConvHx4PktStream, ConcatInt8Stream, HCHUNK, 
                                    TT, TTPARAM, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                                    PADH, PADH, PADW, W1> qLinearConvHx4StreamChunkChunkHPktStream(
  "qLinearConvHx4StreamChunkChunkHPktStream", int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx4StreamChunkHPktStream.txt");


/*
Pad2DStreamInt8<a,3,26,28,32,1,1,1,3> start = 764,end = 3287,total = 2523
SplitFilterInt8PktStream<a,1,2688,896,0>::filter3 start = 772,end = 3304,total = 2532
QLinearConvHx4_0<a,a,28,32,28,32,1,1,1,1,4,3,3,1> start = 1661,end = 4612,total = 2951
QLinearConvHx4_1<a,a,28,32,28,32,1,1,1,1,4,3,3,1> start = 2495,end = 4620,total = 2125
QLinearConvHx4_2<a,a,28,32,28,32,1,1,1,1,4,3,3,1> start = 3323,end = 4643,total = 1320
*/
const int CCHUNK = 1;
QLinearConvChunkCGraphTest<QLinearConvHx4_0, QLinearConvHx4_1, QLinearConvHx4_2, ConcatInt8Stream, CCHUNK, 
                           TT, TTPARAM, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, KH, KW, GROUP, 
                           PADH, PADH, PADW, W1> qLinearConvHx4StreamChunkC(
  "qLinearConvHx4StreamChunkC", int8weights_3x3_int16int8mac, int8bias_3x3, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_3x3_shape1x4x26x28_QLinearConvHx4StreamChunkC.txt");


std::vector<TTPARAM> int8weights_1x1_pad {0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
std::vector<int32_t> int8bias_1x1 {-75, -134, -192, -250};
const int W1_1x1 = (INP_W + 15)/16*16 - INP_W;
QLinearConvChunkHPktStreamGraphTest<QLinearConv1x1StreamInputPackets, ConcatInt8Stream, 13, 
                                    TT, TTPARAM, INP_H, INP_W, INP_W_PAD16, OUT_W, OUT_W_PAD16, STEP_H, STEP_W, B, C, M, 1, 1, GROUP, 
                                    0, 0, 0, W1_1x1> qLinearConv1x1PktStreamChunkHPktStream(
  "qLinearConv1x1PktStreamChunkHPktStream", int8bias_1x1, 0.004, 0.003, 0.002, 25, 0, 19,
  "qlinearconv_int8in_pad.txt", "qlinearconv_int8out_1x1_shape1x4x26x28_QLinearConvScalarStreamHPktStream.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // 3x3 stride 1
  int int8weights_3x3_int16int8mac_size = M*C*16 * sizeof(TT);
  TT* int8weights_3x3_int16int8mac_buf = (TT *) adf::GMIO::malloc(int8weights_3x3_int16int8mac_size);
  memcpy(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac.data(), int8weights_3x3_int16int8mac_size);

  adfCheck(qLinearConvHx4Stream.init(), "init qLinearConvHx4Stream");
  qLinearConvHx4Stream.gmio_w].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4Stream.run(ITER_CNT), "run qLinearConvHx4Stream");
	adfCheck(qLinearConvHx4Stream.end(), "end qLinearConvHx4Stream");

  adfCheck(qLinearConvHx4StreamChunkChunkH.init(), "init qLinearConvHx4StreamChunkChunkH");
  qLinearConvHx4StreamChunkChunkH.gmio_w].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4StreamChunkChunkH.run(ITER_CNT), "run qLinearConvHx4StreamChunkChunkH");
	adfCheck(qLinearConvHx4StreamChunkChunkH.end(), "end qLinearConvHx4StreamChunkChunkH");

  adfCheck(qLinearConvHx4StreamChunkChunkHPktStream.init(), "init qLinearConvHx4StreamChunkChunkHPktStream");
  qLinearConvHx4StreamChunkChunkHPktStream.gmio_w].gm2aie_nb(int8weights_3x3_int16int8mac_buf, int8weights_3x3_int16int8mac_size);
  adfCheck(qLinearConvHx4StreamChunkChunkHPktStream.run(ITER_CNT), "run qLinearConvHx4StreamChunkChunkHPktStream");
	adfCheck(qLinearConvHx4StreamChunkChunkHPktStream.end(), "end qLinearConvHx4StreamChunkChunkHPktStream");

  adfCheck(qLinearConvHx4StreamChunkC.init(), "init qLinearConvHx4StreamChunkC");
  adfCheck(qLinearConvHx4StreamChunkC.run(ITER_CNT), "run qLinearConvHx4StreamChunkC");
	adfCheck(qLinearConvHx4StreamChunkC.end(), "end qLinearConvHx4StreamChunkC");

  adf::GMIO::free(int8weights_3x3_int16int8mac_buf);

  // 1x1 stride 1
  int int8weights_1x1_pad_size = M*((C+15)/16*16) * sizeof(TT);
  TT* int8weights_1x1_pad_buf = (TT *) adf::GMIO::malloc(int8weights_1x1_pad_size);
  memcpy(int8weights_1x1_pad_buf, int8weights_1x1_pad.data(), int8weights_1x1_pad_size);

  adfCheck(qLinearConv1x1PktStreamChunkHPktStream.init(), "init qLinearConv1x1PktStreamChunkHPktStream");
  qLinearConv1x1PktStreamChunkHPktStream.gmio_w].gm2aie_nb(int8weights_1x1_pad_buf, int8weights_1x1_pad_size);
  adfCheck(qLinearConv1x1PktStreamChunkHPktStream.run(ITER_CNT), "run qLinearConv1x1PktStreamChunkHPktStream");
	adfCheck(qLinearConv1x1PktStreamChunkHPktStream.end(), "end qLinearConv1x1PktStreamChunkHPktStream");

  adf::GMIO::free(int8weights_1x1_pad_buf);
  return 0;
}
#endif
