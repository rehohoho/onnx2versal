#include "graph_conv.h"
#include "graph_utils.h"

#define ITER_CNT 1


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


// instance to be compiled and used in host within xclbin
// char empty_input[]  = "empty.txt";
std::vector<float> weights {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
std::vector<float> bias {1, 1};
std::vector<float> weights_rand {0.35798393686119634, 0.4351419865163296, 0.5909267255335692, 0.7223915186982364, 0.31763187327394615, 0.3289537599189831, 0.019691642723703717, 0.040874860094018306, 0.2578216943085364, 0.7402449976749567, 0.6283138303739122, 0.7697890206778347, 0.7689194362148217, 0.8565674676693013, 0.7203192659868836, 0.9790109190008228, 0.8988252193018174, 0.5867171662232342, 0.5881576704911717, 0.034267040352229494, 0.9985265777083543, 0.13157599736614178, 0.740347196631592, 0.8210151951243089, 0.3730545293052032, 0.19685205466531375, 0.09875988679055503, 0.7486060058295778, 0.4526535292056957, 0.7137177590011357, 0.9154076488518006, 0.1465837361510567, 0.9191710007237996, 0.4116264595084367, 0.30526700989728905, 0.9430622606027791, 0.9906516926063994, 0.19889221776744814, 0.6568383469519833, 0.10649531377106036, 0.6509140038575058, 0.8273132277758497, 0.6844985465240676, 0.41733314206259575, 0.38306635956376955, 0.39312241522341707, 0.5897118179929232, 0.8815672700724956, 0.9290661572687678, 0.05352962020731811, 0.1816223946456883, 0.11222431582828851, 0.19333464076691398, 0.3466078106091718, 0.5065316826226238, 0.6294612270091522, 0.7321422191397015, 0.8901115413858071, 0.9890884372297908, 0.6628564785571198, 0.845364518667763, 0.7780388469246989, 0.30753203921871197, 0.875692270234923, 0.042763137947798846, 0.00036734375145786036, 0.2737326293884642, 0.4620975296274499, 0.6383628950006773, 0.1017702665445448, 0.67301013383484, 0.8018158670596945, 0.1853129195793668, 0.41512525483179386, 0.5199849899533098, 0.45180701808448065, 0.7998299308884707, 0.9605223981193308, 0.79895316400818, 0.07799281787582457, 0.8049355721591271, 0.06659633223555117, 0.23597038524157, 0.1530968968357228, 0.19751910684608542, 0.5283151270573506, 0.6716898576108536, 0.470321282336041, 0.9596956390292436, 0.24029232003016665, 0.7631402302547929, 0.8701821785071485, 0.5620661107547191, 0.4562225019615531, 0.5961844467792436, 0.42880977016034294, 0.5551938823233694, 0.4169339521288016, 0.40046971029413403, 0.6953464681395298};
std::vector<float> bias_rand {0.09285121306796795, 0.16654207272882082};

ConvReluGraphTest<ConvReluScalarBCHW, 28, 24, 1, 2, 2, 5> fpscalar_bchw(
  "fpscalar_bchw", weights, bias, "conv_fpin.txt", "conv_fpout_ConvReluScalarBCHW.txt");
ConvReluGraphTest<ConvReluScalarBCHW, 28, 24, 1, 2, 2, 5> fpscalar_bchw_rand(
  "fpscalar_bchw_rand", weights_rand, bias_rand, "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarBCHW_rand.txt");

ConvReluGraphTest<ConvReluScalarBHWC, 28, 24, 1, 2, 2, 5> fpscalar_bhwc(
  "fpscalar_bhwc", weights, bias, "conv_fpin.txt", "conv_fpout_ConvReluScalarBHWC.txt");
ConvReluGraphTest<ConvReluScalarBHWC, 28, 24, 1, 2, 2, 5> fpscalar_bhwc_rand(
  "fpscalar_bhwc_rand", weights_rand, bias, "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarBHWC_rand.txt");

ConvReluGraphTest<Conv5x5ReluBCHW, 28, 24, 1, 2, 2, 5> fpvector_bchw(
  "fpvector_bchw", weights, bias, "conv_fpin.txt", "conv_fpout_Conv5x5ReluBCHW.txt");
ConvReluGraphTest<Conv5x5ReluBCHW, 28, 24, 1, 2, 2, 5> fpvector_bchw_rand(
  "fpvector_bchw_rand", weights_rand, bias_rand, "conv_fpin_rand.txt", "conv_fpout_Conv5x5ReluBCHW_rand.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(fpscalar_bchw.init(), "init fpscalar_bchw");
  adfCheck(fpscalar_bchw.run(1), "run fpscalar_bchw");
	adfCheck(fpscalar_bchw.end(), "end fpscalar_bchw");
  adfCheck(fpscalar_bchw_rand.init(), "init fpscalar_bchw_rand");
  adfCheck(fpscalar_bchw_rand.run(1), "run fpscalar_bchw_rand");
	adfCheck(fpscalar_bchw_rand.end(), "end fpscalar_bchw_rand");

  adfCheck(fpscalar_bhwc.init(), "init fpscalar_bhwc");
  adfCheck(fpscalar_bhwc.run(1), "run fpscalar_bhwc");
	adfCheck(fpscalar_bhwc.end(), "end fpscalar_bhwc");
  adfCheck(fpscalar_bhwc_rand.init(), "init fpscalar_bhwc_rand");
  adfCheck(fpscalar_bhwc_rand.run(1), "run fpscalar_bhwc_rand");
	adfCheck(fpscalar_bhwc_rand.end(), "end fpscalar_bhwc_rand");

  adfCheck(fpvector_bchw.init(), "init fpvector_bchw");
  adfCheck(fpvector_bchw.run(1), "run fpvector_bchw");
	adfCheck(fpvector_bchw.end(), "end fpvector_bchw");
  adfCheck(fpvector_bchw_rand.init(), "init fpvector_bchw_rand");
  adfCheck(fpvector_bchw_rand.run(1), "run fpvector_bchw_rand");
	adfCheck(fpvector_bchw_rand.end(), "end fpvector_bchw_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  adfCheck(fpscalar_bchw.init(), "init fpscalar_bchw");
  get_graph_throughput_by_port(fpscalar_bchw, "plout[0]", fpscalar_bchw.plout[0], 1*8*8*8, sizeof(float_t), ITER_CNT);
	adfCheck(fpscalar_bchw.end(), "end fpscalar_bchw");

  adfCheck(fpscalar_bchw_rand.init(), "init fpscalar_bchw_rand");
  get_graph_throughput_by_port(fpscalar_bchw_rand, "plout[0]", fpscalar_bchw_rand.plout[0], 1*8*8*8, sizeof(float_t), ITER_CNT);
	adfCheck(fpscalar_bchw_rand.end(), "end fpscalar_bchw_rand");


  adfCheck(fpscalar_bhwc.init(), "init fpscalar_bhwc");
  get_graph_throughput_by_port(fpscalar_bhwc, "plout[0]", fpscalar_bhwc.plout[0], 1*8*8*8, sizeof(float_t), ITER_CNT);
	adfCheck(fpscalar_bhwc.end(), "end fpscalar_bhwc");

  adfCheck(fpscalar_bhwc_rand.init(), "init fpscalar_bhwc_rand");
  get_graph_throughput_by_port(fpscalar_bhwc_rand, "plout[0]", fpscalar_bhwc_rand.plout[0], 1*8*8*8, sizeof(float_t), ITER_CNT);
	adfCheck(fpscalar_bhwc_rand.end(), "end fpscalar_bhwc_rand");
  

  adfCheck(fpvector_bchw.init(), "init fpvector_bchw");
  get_graph_throughput_by_port(fpvector_bchw, "plout[0]", fpvector_bchw.plout[0], 1*8*8*8, sizeof(float_t), ITER_CNT);
	adfCheck(fpvector_bchw.end(), "end fpvector_bchw");

  adfCheck(fpvector_bchw_rand.init(), "init fpvector_bchw_rand");
  get_graph_throughput_by_port(fpvector_bchw_rand, "plout[0]", fpvector_bchw_rand.plout[0], 1*8*8*8, sizeof(float_t), ITER_CNT);
	adfCheck(fpvector_bchw_rand.end(), "end fpvector_bchw_rand");
  return 0;
}
#endif
