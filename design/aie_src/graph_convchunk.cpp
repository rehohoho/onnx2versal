#include "graph_conv.h"
#include "graph_utils.h"


template <
  template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int IS_BCHW, int MCHUNK, 
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkMGraphTest : public adf::graph {

  private:
    typedef ConvReluChunkMGraph<CONV, CONCAT, IS_BCHW, MCHUNK, 
                                INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                B, C, M, KH, KW, GROUP, IS_RELU,
                                H0, H1, W0, W1> Graph;
    Graph g;
    static constexpr int OUT_H = Graph::OUT_H;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    ConvReluChunkMGraphTest(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(weights, bias) { 
      plin[0] = adf::input_plio::create("plin0_conv"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_conv"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*C*INP_H*INP_W*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


template <
  template<typename, int, int, int, int> class SPLIT,
  template<int, int, int, int, int, int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  int INP_H, int INP_W, int OUT_W, int OUT_W_PAD, int STEP_H, int STEP_W,
  int B, int C, int M, int KH, int KW, int GROUP, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkHGraphTest : public adf::graph {

  private:
    typedef ConvReluChunkHGraph<SPLIT, CONV, CONCAT, HCHUNK, 
                                INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W,
                                B, C, M, KH, KW, GROUP, IS_RELU,
                                H0, H1, W0, W1> Graph;
    Graph g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];
    adf::input_gmio gmio_w;

    ConvReluChunkHGraphTest(
      const std::string& id,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(bias) { 
      plin[0] = adf::input_plio::create("plin0_conv"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_conv"+id+"_output", PLIO64_ARG(OUT_TXT));
      gmio_w = adf::input_gmio::create("gmio0_gemm"+id+"_w", 64, 1000);
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (gmio_w.out[0], g.pin[1]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
const int INP_H = 24;
const int INP_W = 24;
const int OUT_W = INP_W;
const int OUT_W_PAD = (INP_W + 3)/4*4;
const int STEP_H = 1;
const int STEP_W = 1;
const int B = 1;
const int C = 1; // loop dependency missed issue occurs at C=1
const int M = 4;
const int KH = 5;
const int KW = 5;
const int GROUP = 1;
const int IS_RELU = 1;
const int MCHUNK = 2; // aiecompiler have issues when this is 1 (deadlock?) or 3 (invalid seg)
const int HCHUNK = 8; // OUT_H' * strides + overlap, overlap = KH - strides
const int PADH = KH/2;
const int PADW = KW/2;

std::vector<float> fpweights {-0.33953237533569336, 0.38630467653274536, -0.0536055862903595, 0.40787559747695923, -0.33976954221725464, 0.16111749410629272, -0.05973625183105469, -0.4235132336616516, 0.19646316766738892, -0.2526012659072876, -0.4603844881057739, -0.4400556981563568, -0.4389214515686035, 0.4077329635620117, 0.2398838996887207, 0.3980623483657837, 0.17258232831954956, 0.028939902782440186, -0.19555363059043884, 0.49796223640441895, -0.13781094551086426, -0.02935105562210083, -0.12175482511520386, 0.47952693700790405, -0.3253416121006012, -0.17201200127601624, 0.1803486943244934, -0.43679237365722656, 0.10724937915802002, -0.02235350012779236, -0.2160000205039978, -0.2615867257118225, 0.014512717723846436, -0.13207241892814636, -0.043480098247528076, -0.16252261400222778, 0.4704936742782593, -0.36656057834625244, -0.4031960368156433, -0.15660828351974487, 0.0910269021987915, 0.15917646884918213, -0.1027432382106781, 0.4992780089378357, -0.14810699224472046, 0.22140663862228394, 0.1375827193260193, 0.31305384635925293, 0.47622567415237427, 0.38979363441467285, 0.2645619511604309, 0.1982485055923462, -0.16450181603431702, -0.35231441259384155, -0.43736398220062256, -0.258098304271698, -0.067718505859375, 0.021996259689331055, 0.27308356761932373, 0.4587409496307373, -0.3826795220375061, -0.3929958641529083, 0.08969473838806152, 0.24539804458618164, 0.348150372505188, 0.43583208322525024, 0.48342621326446533, -0.10019829869270325, -0.11966481804847717, -0.35219132900238037, 0.18493443727493286, 0.15676194429397583, 0.36206257343292236, -0.4027419984340668, -0.002223104238510132, 0.08108192682266235, -0.2584429383277893, -0.3309745788574219, 0.3595808148384094, -0.4414650797843933, -0.02937909960746765, -0.38416600227355957, -0.04294124245643616, 0.4799623489379883, -0.07629364728927612, 0.357124924659729, -0.3826844394207001, -0.22874793410301208, -0.09620726108551025, -0.10018786787986755, 0.17138350009918213, -0.15528187155723572, 0.21376687288284302, 0.13918691873550415, -0.10083884000778198, -0.06823986768722534, 0.11452770233154297, -0.4299578070640564, 0.32240670919418335, 0.15342116355895996};
std::vector<float> fpweights_pad {-0.33953237533569336, 0.38630467653274536, -0.0536055862903595, 0.40787559747695923, -0.33976954221725464, 0.0, 0.0, 0.0, 0.16111749410629272, -0.05973625183105469, -0.4235132336616516, 0.19646316766738892, -0.2526012659072876, 0.0, 0.0, 0.0, -0.4603844881057739, -0.4400556981563568, -0.4389214515686035, 0.4077329635620117, 0.2398838996887207, 0.0, 0.0, 0.0, 0.3980623483657837, 0.17258232831954956, 0.028939902782440186, -0.19555363059043884, 0.49796223640441895, 0.0, 0.0, 0.0, -0.13781094551086426, -0.02935105562210083, -0.12175482511520386, 0.47952693700790405, -0.3253416121006012, 0.0, 0.0, 0.0, -0.17201200127601624, 0.1803486943244934, -0.43679237365722656, 0.10724937915802002, -0.02235350012779236, 0.0, 0.0, 0.0, -0.2160000205039978, -0.2615867257118225, 0.014512717723846436, -0.13207241892814636, -0.043480098247528076, 0.0, 0.0, 0.0, -0.16252261400222778, 0.4704936742782593, -0.36656057834625244, -0.4031960368156433, -0.15660828351974487, 0.0, 0.0, 0.0, 0.0910269021987915, 0.15917646884918213, -0.1027432382106781, 0.4992780089378357, -0.14810699224472046, 0.0, 0.0, 0.0, 0.22140663862228394, 0.1375827193260193, 0.31305384635925293, 0.47622567415237427, 0.38979363441467285, 0.0, 0.0, 0.0, 0.2645619511604309, 0.1982485055923462, -0.16450181603431702, -0.35231441259384155, -0.43736398220062256, 0.0, 0.0, 0.0, -0.258098304271698, -0.067718505859375, 0.021996259689331055, 0.27308356761932373, 0.4587409496307373, 0.0, 0.0, 0.0, -0.3826795220375061, -0.3929958641529083, 0.08969473838806152, 0.24539804458618164, 0.348150372505188, 0.0, 0.0, 0.0, 0.43583208322525024, 0.48342621326446533, -0.10019829869270325, -0.11966481804847717, -0.35219132900238037, 0.0, 0.0, 0.0, 0.18493443727493286, 0.15676194429397583, 0.36206257343292236, -0.4027419984340668, -0.002223104238510132, 0.0, 0.0, 0.0, 0.08108192682266235, -0.2584429383277893, -0.3309745788574219, 0.3595808148384094, -0.4414650797843933, 0.0, 0.0, 0.0, -0.02937909960746765, -0.38416600227355957, -0.04294124245643616, 0.4799623489379883, -0.07629364728927612, 0.0, 0.0, 0.0, 0.357124924659729, -0.3826844394207001, -0.22874793410301208, -0.09620726108551025, -0.10018786787986755, 0.0, 0.0, 0.0, 0.17138350009918213, -0.15528187155723572, 0.21376687288284302, 0.13918691873550415, -0.10083884000778198, 0.0, 0.0, 0.0, -0.06823986768722534, 0.11452770233154297, -0.4299578070640564, 0.32240670919418335, 0.15342116355895996, 0.0, 0.0, 0.0};
std::vector<float> fpbias {0.22634243965148926, 0.036922991275787354, -0.38952288031578064, -0.09496438503265381};

// BCHW
ConvReluChunkMGraphTest<ConvReluScalarBCHW, ConcatFloat, 1, MCHUNK, 
                        INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                        B, C, M, KH, KW, GROUP, IS_RELU,
                        PADH, PADH, PADW, PADW> convReluScalarBCHW(
  "convReluScalarBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x4x24x24_ConvReluScalarBCHW.txt");

ConvReluChunkMGraphTest<Conv5x5ReluBCHW, ConcatFloat, 1, MCHUNK, 
                        INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                        B, C, M, KH, KW, GROUP, IS_RELU,
                        PADH, PADH, PADW, PADW> conv5x5ReluBCHW(
  "conv5x5ReluBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x4x24x24_Conv5x5ReluBCHW.txt");

ConvReluChunkMGraphTest<Conv5x5on8ReluBCHW, ConcatFloat, 1, MCHUNK, 
                        INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                        B, C, M, KH, KW, GROUP, IS_RELU,
                        PADH, PADH, PADW, PADW> conv5x5on8ReluBCHW(
  "conv5x5on8ReluBCHW", fpweights_pad, fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x4x24x24_Conv5x5on8ReluBCHW.txt");

ConvReluChunkHGraphTest<SplitScalar, ConvReluScalarStreamCacheCKK, ConcatScalarStream, HCHUNK, 
                        INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                        B, C, M, KH, KW, GROUP, IS_RELU,
                        PADH, PADH, PADW, PADW> convReluScalarStreamCacheCKK(
  "convReluScalarStreamCacheCKK", fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x4x24x24_ConvReluScalarStreamCacheCKK.txt");

// BHWC, ConcatFloat requires CONCAT_BLOCK=M%4=0
ConvReluChunkMGraphTest<ConvReluScalarBHWC, ConcatScalar, 0, MCHUNK, 
                        INP_H, INP_W, OUT_W, OUT_W_PAD, STEP_H, STEP_W, 
                        B, C, M, KH, KW, GROUP, IS_RELU,
                        PADH, PADH, PADW, PADW> convReluScalarBHWC(
  "convReluScalarBHWC", fpweights, fpbias, 
  "conv_fpin.txt", "convbhwc_fpout_shape1x24x24x4_ConvReluScalarBHWC.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  // init gmio
  int mckk_bufsize = M*C*KH*KW * sizeof(float_t);
  float_t* mckk_buf = (float_t *) adf::GMIO::malloc(mckk_bufsize);
  memcpy(mckk_buf, fpweights.data(), mckk_bufsize);

  // BCHW
  adfCheck(convReluScalarBCHW.init(), "init convReluScalarBCHW");
  adfCheck(convReluScalarBCHW.run(ITER_CNT), "run convReluScalarBCHW");
	adfCheck(convReluScalarBCHW.end(), "end convReluScalarBCHW");

  adfCheck(conv5x5ReluBCHW.init(), "init conv5x5ReluBCHW");
  adfCheck(conv5x5ReluBCHW.run(ITER_CNT), "run conv5x5ReluBCHW");
	adfCheck(conv5x5ReluBCHW.end(), "end conv5x5ReluBCHW");

  adfCheck(conv5x5on8ReluBCHW.init(), "init conv5x5on8ReluBCHW");
  adfCheck(conv5x5on8ReluBCHW.run(ITER_CNT), "run conv5x5on8ReluBCHW");
	adfCheck(conv5x5on8ReluBCHW.end(), "end conv5x5on8ReluBCHW");

  adfCheck(convReluScalarStreamCacheCKK.init(), "init convReluScalarStreamCacheCKK");
  convReluScalarStreamCacheCKK.gmio_w.gm2aie_nb(mckk_buf, mckk_bufsize);
  adfCheck(convReluScalarStreamCacheCKK.run(ITER_CNT), "run convReluScalarStreamCacheCKK");
	adfCheck(convReluScalarStreamCacheCKK.end(), "end convReluScalarStreamCacheCKK");

  // BHWC
  adfCheck(convReluScalarBHWC.init(), "init convReluScalarBHWC");
  adfCheck(convReluScalarBHWC.run(ITER_CNT), "run convReluScalarBHWC");
	adfCheck(convReluScalarBHWC.end(), "end convReluScalarBHWC");

  // cleanup gmio
  adf::GMIO::free(mckk_buf);
  return 0;
}
#endif
