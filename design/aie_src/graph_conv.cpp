#include "graph_conv.h"
#include "graph_utils.h"


template <template<int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluGraphTest : public adf::graph {

  private:
    ConvReluGraph<CONV, INP_W, OUT_W, B, C, M, K> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    ConvReluGraphTest(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(weights, bias) { 
      plin[0] = adf::input_plio::create("plin0_conv"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_conv"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (g.pout[0], plout[0].in[0]);
    }
};


template <template<int, int, int, int, int, int> class CONV, 
  int INP_W, int OUT_W, int B, int C, int M, int K>
class ConvReluGmemParamGraphTest : public adf::graph {

  private:
    ConvReluGmemParamGraph<CONV, INP_W, OUT_W, B, C, M, K> g;

  public:
    adf::input_plio plin[3];
    adf::output_plio plout[1];

    ConvReluGmemParamGraphTest(
      const std::string& id,
      const std::string& WEIGHTS_TXT,
      const std::string& BIAS_TXT,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_conv"+id+"_input", PLIO64_ARG(INP_TXT));
      plin[1] = adf::input_plio::create("plin1_conv"+id+"_weights", PLIO64_ARG(WEIGHTS_TXT));
      plin[2] = adf::input_plio::create("plin2_conv"+id+"_bias", PLIO64_ARG(BIAS_TXT));
      plout[0] = adf::output_plio::create("plout0_conv"+id+"_output", PLIO64_ARG(OUT_TXT));
      
      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<M*K*K*C*4>>         (plin[1].out[0], g.pin[1]);
      adf::connect<adf::window<M*4>>               (plin[2].out[0], g.pin[2]);
      adf::connect<adf::window<B*OUT_W*OUT_W*M*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
std::vector<float> fpweights {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149};
std::vector<float> fpweights_pad {0, 1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 9, 0, 0, 0, 10, 11, 12, 13, 14, 0, 0, 0, 15, 16, 17, 18, 19, 0, 0, 0, 20, 21, 22, 23, 24, 0, 0, 0, 25, 26, 27, 28, 29, 0, 0, 0, 30, 31, 32, 33, 34, 0, 0, 0, 35, 36, 37, 38, 39, 0, 0, 0, 40, 41, 42, 43, 44, 0, 0, 0, 45, 46, 47, 48, 49, 0, 0, 0, 50, 51, 52, 53, 54, 0, 0, 0, 55, 56, 57, 58, 59, 0, 0, 0, 60, 61, 62, 63, 64, 0, 0, 0, 65, 66, 67, 68, 69, 0, 0, 0, 70, 71, 72, 73, 74, 0, 0, 0, 75, 76, 77, 78, 79, 0, 0, 0, 80, 81, 82, 83, 84, 0, 0, 0, 85, 86, 87, 88, 89, 0, 0, 0, 90, 91, 92, 93, 94, 0, 0, 0, 95, 96, 97, 98, 99, 0, 0, 0, 100, 101, 102, 103, 104, 0, 0, 0, 105, 106, 107, 108, 109, 0, 0, 0, 110, 111, 112, 113, 114, 0, 0, 0, 115, 116, 117, 118, 119, 0, 0, 0, 120, 121, 122, 123, 124, 0, 0, 0, 125, 126, 127, 128, 129, 0, 0, 0, 130, 131, 132, 133, 134, 0, 0, 0, 135, 136, 137, 138, 139, 0, 0, 0, 140, 141, 142, 143, 144, 0, 0, 0, 145, 146, 147, 148, 149, 0, 0, 0};
std::vector<float> fpbias {1, 1, 1, 1, 1, 1};

std::vector<float> fpweights_rand {0.1024137584524164, 0.156383348867963, 0.3041986915994078, 0.07535906908334034, 0.4246630028405929, 0.10761770514958191, 0.568217593669845, 0.24655693981115612, 0.5964330653496227, 0.11752564290363765, 0.9758838684185334, 0.9325612038573404, 0.3917969385646658, 0.24217859412608544, 0.2503982128535728, 0.48339353520239203, 0.03999280190071697, 0.639705106075127, 0.4083029083397448, 0.3774065725888873, 0.8093649714891984, 0.70903546018329, 0.954333815392692, 0.3519362404956907, 0.8975427646494055, 0.7699671862500889, 0.35742465159471304, 0.6216654364532578, 0.2885699576516956, 0.8743999170748423, 0.11242731721231125, 0.21243436129404103, 0.18303329207992114, 0.40302600240428865, 0.7452329600321291, 0.5269074490521803, 0.48767632353820756, 0.0005459648969956543, 0.4254017253550547, 0.06355377483615843, 0.20825325212148382, 0.9323939389604944, 0.2153982043432382, 0.8583376386342625, 0.8028933715613342, 0.15914623694224284, 0.6057119572702788, 0.11566187190501331, 0.7278881583695115, 0.6374622773722066, 0.8119385616910193, 0.47938454938918806, 0.9148630878333829, 0.04934894678843971, 0.29288856502701466, 0.715052597465167, 0.41810921174800086, 0.17295135427115638, 0.10721074542854603, 0.8173391114616214, 0.47314297846564424, 0.8822836719191074, 0.733289134316726, 0.4097262056307436, 0.37351101415568366, 0.5156383466512517, 0.8890599531897286, 0.7372785797141679, 0.00515296426902323, 0.6941578513691256, 0.9195074069058207, 0.7104557595044916, 0.1770057815674959, 0.4835181274274587, 0.1403160179234194, 0.3589952783396321, 0.9371170419405177, 0.9233053075587083, 0.2828368521760829, 0.33963104416619916, 0.6002128681312939, 0.96319729526038, 0.14780133406539042, 0.2569166436866691, 0.8735568272907714, 0.4918922317083445, 0.8989610922270317, 0.18551789752317627, 0.5326685874713607, 0.32626963264937237, 0.31654255989247604, 0.44687696394619913, 0.43307744910126844, 0.3573468796779544, 0.9149707703156186, 0.7317441854328928, 0.7275469913315297, 0.2899134495919554, 0.5777094243168404, 0.779179433301834, 0.7955903685432131, 0.34453046075431226, 0.7708727565686478, 0.735893896807733, 0.14150648562190027, 0.8659454685664772, 0.4413214701804108, 0.48641044888866547, 0.4483691788979973, 0.5678460014775075, 0.6211692473670547, 0.4981795657629434, 0.8667885432590956, 0.6277347561952844, 0.40142794930551995, 0.41669175690871096, 0.8108386151289514, 0.3481919427465201, 0.21145479578241355, 0.059383188005789234, 0.8760268479205742, 0.9185464511903499, 0.12012018216347597, 0.33447374149611486, 0.17537206951524387, 0.11589846882587973, 0.8998667430000302, 0.05687725914535546, 0.9804856634690068, 0.09645086069738418, 0.8634706491935857, 0.5665061069891627, 0.36791748781787337, 0.3423423766251579, 0.7573641432377087, 0.3145732950042872, 0.6573189166171418, 0.5173260835160801, 0.4849656451580705, 0.9011621706491616, 0.5546450586202596, 0.8268616030486949, 0.7255735341014894, 0.03855724605899835, 0.7731100525054192, 0.21687025009104066, 0.9031496468515715, 0.042924190608832014, 0.33307203447431877, 0.09973294723475401};
std::vector<float> fpweights_rand_pad {0.1024137584524164, 0.156383348867963, 0.3041986915994078, 0.07535906908334034, 0.4246630028405929, 0.0, 0.0, 0.0, 0.10761770514958191, 0.568217593669845, 0.24655693981115612, 0.5964330653496227, 0.11752564290363765, 0.0, 0.0, 0.0, 0.9758838684185334, 0.9325612038573404, 0.3917969385646658, 0.24217859412608544, 0.2503982128535728, 0.0, 0.0, 0.0, 0.48339353520239203, 0.03999280190071697, 0.639705106075127, 0.4083029083397448, 0.3774065725888873, 0.0, 0.0, 0.0, 0.8093649714891984, 0.70903546018329, 0.954333815392692, 0.3519362404956907, 0.8975427646494055, 0.0, 0.0, 0.0, 0.7699671862500889, 0.35742465159471304, 0.6216654364532578, 0.2885699576516956, 0.8743999170748423, 0.0, 0.0, 0.0, 0.11242731721231125, 0.21243436129404103, 0.18303329207992114, 0.40302600240428865, 0.7452329600321291, 0.0, 0.0, 0.0, 0.5269074490521803, 0.48767632353820756, 0.0005459648969956543, 0.4254017253550547, 0.06355377483615843, 0.0, 0.0, 0.0, 0.20825325212148382, 0.9323939389604944, 0.2153982043432382, 0.8583376386342625, 0.8028933715613342, 0.0, 0.0, 0.0, 0.15914623694224284, 0.6057119572702788, 0.11566187190501331, 0.7278881583695115, 0.6374622773722066, 0.0, 0.0, 0.0, 0.8119385616910193, 0.47938454938918806, 0.9148630878333829, 0.04934894678843971, 0.29288856502701466, 0.0, 0.0, 0.0, 0.715052597465167, 0.41810921174800086, 0.17295135427115638, 0.10721074542854603, 0.8173391114616214, 0.0, 0.0, 0.0, 0.47314297846564424, 0.8822836719191074, 0.733289134316726, 0.4097262056307436, 0.37351101415568366, 0.0, 0.0, 0.0, 0.5156383466512517, 0.8890599531897286, 0.7372785797141679, 0.00515296426902323, 0.6941578513691256, 0.0, 0.0, 0.0, 0.9195074069058207, 0.7104557595044916, 0.1770057815674959, 0.4835181274274587, 0.1403160179234194, 0.0, 0.0, 0.0, 0.3589952783396321, 0.9371170419405177, 0.9233053075587083, 0.2828368521760829, 0.33963104416619916, 0.0, 0.0, 0.0, 0.6002128681312939, 0.96319729526038, 0.14780133406539042, 0.2569166436866691, 0.8735568272907714, 0.0, 0.0, 0.0, 0.4918922317083445, 0.8989610922270317, 0.18551789752317627, 0.5326685874713607, 0.32626963264937237, 0.0, 0.0, 0.0, 0.31654255989247604, 0.44687696394619913, 0.43307744910126844, 0.3573468796779544, 0.9149707703156186, 0.0, 0.0, 0.0, 0.7317441854328928, 0.7275469913315297, 0.2899134495919554, 0.5777094243168404, 0.779179433301834, 0.0, 0.0, 0.0, 0.7955903685432131, 0.34453046075431226, 0.7708727565686478, 0.735893896807733, 0.14150648562190027, 0.0, 0.0, 0.0, 0.8659454685664772, 0.4413214701804108, 0.48641044888866547, 0.4483691788979973, 0.5678460014775075, 0.0, 0.0, 0.0, 0.6211692473670547, 0.4981795657629434, 0.8667885432590956, 0.6277347561952844, 0.40142794930551995, 0.0, 0.0, 0.0, 0.41669175690871096, 0.8108386151289514, 0.3481919427465201, 0.21145479578241355, 0.059383188005789234, 0.0, 0.0, 0.0, 0.8760268479205742, 0.9185464511903499, 0.12012018216347597, 0.33447374149611486, 0.17537206951524387, 0.0, 0.0, 0.0, 0.11589846882587973, 0.8998667430000302, 0.05687725914535546, 0.9804856634690068, 0.09645086069738418, 0.0, 0.0, 0.0, 0.8634706491935857, 0.5665061069891627, 0.36791748781787337, 0.3423423766251579, 0.7573641432377087, 0.0, 0.0, 0.0, 0.3145732950042872, 0.6573189166171418, 0.5173260835160801, 0.4849656451580705, 0.9011621706491616, 0.0, 0.0, 0.0, 0.5546450586202596, 0.8268616030486949, 0.7255735341014894, 0.03855724605899835, 0.7731100525054192, 0.0, 0.0, 0.0, 0.21687025009104066, 0.9031496468515715, 0.042924190608832014, 0.33307203447431877, 0.09973294723475401, 0.0, 0.0, 0.0};
std::vector<float> fpbias_rand {0.47558911708484375, 0.8200224358697518, 0.2981873596630641, 0.1509348973110416, 0.3302670356968992, 0.813880141920636};

//BCHW
ConvReluGraphTest<ConvReluScalarBCHW, 28, 24, 1, 1, 6, 5> convReluScalarBCHW(
  "convReluScalarBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "conv_fpout_ConvReluScalarBCHW.txt");
ConvReluGraphTest<ConvReluScalarBCHW, 28, 24, 1, 1, 6, 5> convReluScalarBCHW_rand(
  "convReluScalarBCHW_rand", fpweights_rand, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarBCHW_rand.txt");

ConvReluGraphTest<Conv5x5ReluBCHW, 28, 24, 1, 1, 6, 5> conv5x5ReluBCHW(
  "conv5x5ReluBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "conv_fpout_Conv5x5ReluBCHW.txt");
ConvReluGraphTest<Conv5x5ReluBCHW, 28, 24, 1, 1, 6, 5> conv5x5ReluBCHW_rand(
  "conv5x5ReluBCHW_rand", fpweights_rand, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_Conv5x5ReluBCHW_rand.txt");

ConvReluGraphTest<Conv5x5on8ReluBCHW, 28, 24, 1, 1, 6, 5> conv5x5on8ReluBCHW(
  "conv5x5on8ReluBCHW", fpweights_pad, fpbias, 
  "conv_fpin.txt", "conv_fpout_Conv5x5on8ReluBCHW.txt");
ConvReluGraphTest<Conv5x5on8ReluBCHW, 28, 24, 1, 1, 6, 5> conv5x5on8ReluBCHW_rand(
  "conv5x5on8ReluBCHW_rand", fpweights_rand_pad, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_Conv5x5on8ReluBCHW_rand.txt");

//BHWC
ConvReluGraphTest<ConvReluScalarBHWC, 28, 24, 1, 1, 6, 5> convReluScalarBHWC(
  "convReluScalarBHWC", fpweights, fpbias, 
  "conv_fpin.txt", "conv_fpout_ConvReluScalarBHWC.txt");
ConvReluGraphTest<ConvReluScalarBHWC, 28, 24, 1, 1, 6, 5> convReluScalarBHWC_rand(
  "convReluScalarBHWC_rand", fpweights_rand, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarBHWC_rand.txt");

ConvReluGmemParamGraphTest<ConvReluScalarGmemParamBHWC, 28, 24, 1, 1, 6, 5> convReluScalarGmemParamBHWC(
  "convReluScalarGmemParamBHWC", "conv_fpweights.txt", "conv_fpbias.txt", 
  "conv_fpin.txt", "conv_fpout_ConvReluScalarGmemParamBHWC.txt");
ConvReluGmemParamGraphTest<ConvReluScalarGmemParamBHWC, 28, 24, 1, 1, 6, 5> convReluScalarGmemParamBHWC_rand(
  "convReluScalarGmemParamBHWC_rand", "conv_fpweights_rand.txt", "conv_fpbias_rand.txt", 
  "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarGmemParamBHWC_rand.txt");



#ifdef __X86SIM__
int main(int argc, char ** argv) {
  // BCHW
  adfCheck(convReluScalarBCHW.init(), "init convReluScalarBCHW");
  adfCheck(convReluScalarBCHW.run(ITER_CNT), "run convReluScalarBCHW");
	adfCheck(convReluScalarBCHW.end(), "end convReluScalarBCHW");
  adfCheck(convReluScalarBCHW_rand.init(), "init convReluScalarBCHW_rand");
  adfCheck(convReluScalarBCHW_rand.run(ITER_CNT), "run convReluScalarBCHW_rand");
	adfCheck(convReluScalarBCHW_rand.end(), "end convReluScalarBCHW_rand");

  adfCheck(conv5x5ReluBCHW.init(), "init conv5x5ReluBCHW");
  adfCheck(conv5x5ReluBCHW.run(ITER_CNT), "run conv5x5ReluBCHW");
	adfCheck(conv5x5ReluBCHW.end(), "end conv5x5ReluBCHW");
  adfCheck(conv5x5ReluBCHW_rand.init(), "init conv5x5ReluBCHW_rand");
  adfCheck(conv5x5ReluBCHW_rand.run(ITER_CNT), "run conv5x5ReluBCHW_rand");
	adfCheck(conv5x5ReluBCHW_rand.end(), "end conv5x5ReluBCHW_rand");

  adfCheck(conv5x5on8ReluBCHW.init(), "init conv5x5on8ReluBCHW");
  adfCheck(conv5x5on8ReluBCHW.run(ITER_CNT), "run conv5x5on8ReluBCHW");
	adfCheck(conv5x5on8ReluBCHW.end(), "end conv5x5on8ReluBCHW");
  adfCheck(conv5x5on8ReluBCHW_rand.init(), "init conv5x5on8ReluBCHW_rand");
  adfCheck(conv5x5on8ReluBCHW_rand.run(ITER_CNT), "run conv5x5on8ReluBCHW_rand");
	adfCheck(conv5x5on8ReluBCHW_rand.end(), "end conv5x5on8ReluBCHW_rand");

  // BHWC
  adfCheck(convReluScalarBHWC.init(), "init convReluScalarBHWC");
  adfCheck(convReluScalarBHWC.run(ITER_CNT), "run convReluScalarBHWC");
	adfCheck(convReluScalarBHWC.end(), "end convReluScalarBHWC");
  adfCheck(convReluScalarBHWC_rand.init(), "init convReluScalarBHWC_rand");
  adfCheck(convReluScalarBHWC_rand.run(ITER_CNT), "run convReluScalarBHWC_rand");
	adfCheck(convReluScalarBHWC_rand.end(), "end convReluScalarBHWC_rand");

  adfCheck(convReluScalarGmemParamBHWC.init(), "init convReluScalarGmemParamBHWC");
  adfCheck(convReluScalarGmemParamBHWC.run(ITER_CNT), "run convReluScalarGmemParamBHWC");
	adfCheck(convReluScalarGmemParamBHWC.end(), "end convReluScalarGmemParamBHWC");
  adfCheck(convReluScalarGmemParamBHWC_rand.init(), "init convReluScalarGmemParamBHWC_rand");
  adfCheck(convReluScalarGmemParamBHWC_rand.run(ITER_CNT), "run convReluScalarGmemParamBHWC_rand");
	adfCheck(convReluScalarGmemParamBHWC_rand.end(), "end convReluScalarGmemParamBHWC_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  // BCHW
  adfCheck(convReluScalarBCHW.init(), "init convReluScalarBCHW");
  get_graph_throughput_by_port(convReluScalarBCHW, "plout[0]", convReluScalarBCHW.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBCHW.end(), "end convReluScalarBCHW");
  adfCheck(convReluScalarBCHW_rand.init(), "init convReluScalarBCHW_rand");
  get_graph_throughput_by_port(convReluScalarBCHW_rand, "plout[0]", convReluScalarBCHW_rand.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBCHW_rand.end(), "end convReluScalarBCHW_rand");
  

  adfCheck(conv5x5ReluBCHW.init(), "init conv5x5ReluBCHW");
  get_graph_throughput_by_port(conv5x5ReluBCHW, "plout[0]", conv5x5ReluBCHW.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5ReluBCHW.end(), "end conv5x5ReluBCHW");
  adfCheck(conv5x5ReluBCHW_rand.init(), "init conv5x5ReluBCHW_rand");
  get_graph_throughput_by_port(conv5x5ReluBCHW_rand, "plout[0]", conv5x5ReluBCHW_rand.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5ReluBCHW_rand.end(), "end conv5x5ReluBCHW_rand");

  
  adfCheck(conv5x5on8ReluBCHW.init(), "init conv5x5on8ReluBCHW");
  get_graph_throughput_by_port(conv5x5on8ReluBCHW, "plout[0]", conv5x5on8ReluBCHW.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5on8ReluBCHW.end(), "end conv5x5on8ReluBCHW");
  adfCheck(conv5x5on8ReluBCHW_rand.init(), "init conv5x5on8ReluBCHW_rand");
  get_graph_throughput_by_port(conv5x5on8ReluBCHW_rand, "plout[0]", conv5x5on8ReluBCHW_rand.plout[0], 1*4*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5on8ReluBCHW_rand.end(), "end conv5x5on8ReluBCHW_rand");


  // BHWC
  adfCheck(convReluScalarBHWC.init(), "init convReluScalarBHWC");
  get_graph_throughput_by_port(convReluScalarBHWC, "plout[0]", convReluScalarBHWC.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBHWC.end(), "end convReluScalarBHWC");
  adfCheck(convReluScalarBHWC_rand.init(), "init convReluScalarBHWC_rand");
  get_graph_throughput_by_port(convReluScalarBHWC_rand, "plout[0]", convReluScalarBHWC_rand.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBHWC_rand.end(), "end convReluScalarBHWC_rand");


  adfCheck(convReluScalarGmemParamBHWC.init(), "init convReluScalarGmemParamBHWC");
  get_graph_throughput_by_port(convReluScalarGmemParamBHWC, "plout[0]", convReluScalarGmemParamBHWC.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarGmemParamBHWC.end(), "end convReluScalarGmemParamBHWC");
  adfCheck(convReluScalarGmemParamBHWC_rand.init(), "init convReluScalarGmemParamBHWC_rand");
  get_graph_throughput_by_port(convReluScalarGmemParamBHWC_rand, "plout[0]", convReluScalarGmemParamBHWC_rand.plout[0], 1*6*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarGmemParamBHWC_rand.end(), "end convReluScalarGmemParamBHWC_rand");
  return 0;
}
#endif
