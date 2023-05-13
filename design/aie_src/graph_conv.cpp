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
std::vector<float> fpweights {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199};
std::vector<float> fpweights_pad {0, 1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 9, 0, 0, 0, 10, 11, 12, 13, 14, 0, 0, 0, 15, 16, 17, 18, 19, 0, 0, 0, 20, 21, 22, 23, 24, 0, 0, 0, 25, 26, 27, 28, 29, 0, 0, 0, 30, 31, 32, 33, 34, 0, 0, 0, 35, 36, 37, 38, 39, 0, 0, 0, 40, 41, 42, 43, 44, 0, 0, 0, 45, 46, 47, 48, 49, 0, 0, 0, 50, 51, 52, 53, 54, 0, 0, 0, 55, 56, 57, 58, 59, 0, 0, 0, 60, 61, 62, 63, 64, 0, 0, 0, 65, 66, 67, 68, 69, 0, 0, 0, 70, 71, 72, 73, 74, 0, 0, 0, 75, 76, 77, 78, 79, 0, 0, 0, 80, 81, 82, 83, 84, 0, 0, 0, 85, 86, 87, 88, 89, 0, 0, 0, 90, 91, 92, 93, 94, 0, 0, 0, 95, 96, 97, 98, 99, 0, 0, 0, 100, 101, 102, 103, 104, 0, 0, 0, 105, 106, 107, 108, 109, 0, 0, 0, 110, 111, 112, 113, 114, 0, 0, 0, 115, 116, 117, 118, 119, 0, 0, 0, 120, 121, 122, 123, 124, 0, 0, 0, 125, 126, 127, 128, 129, 0, 0, 0, 130, 131, 132, 133, 134, 0, 0, 0, 135, 136, 137, 138, 139, 0, 0, 0, 140, 141, 142, 143, 144, 0, 0, 0, 145, 146, 147, 148, 149, 0, 0, 0, 150, 151, 152, 153, 154, 0, 0, 0, 155, 156, 157, 158, 159, 0, 0, 0, 160, 161, 162, 163, 164, 0, 0, 0, 165, 166, 167, 168, 169, 0, 0, 0, 170, 171, 172, 173, 174, 0, 0, 0, 175, 176, 177, 178, 179, 0, 0, 0, 180, 181, 182, 183, 184, 0, 0, 0, 185, 186, 187, 188, 189, 0, 0, 0, 190, 191, 192, 193, 194, 0, 0, 0, 195, 196, 197, 198, 199, 0, 0, 0};
std::vector<float> fpbias {1, 1, 1, 1};

std::vector<float> fpweights_rand {0.35798393686119634, 0.4351419865163296, 0.5909267255335692, 0.7223915186982364, 0.31763187327394615, 0.3289537599189831, 0.019691642723703717, 0.040874860094018306, 0.2578216943085364, 0.7402449976749567, 0.6283138303739122, 0.7697890206778347, 0.7689194362148217, 0.8565674676693013, 0.7203192659868836, 0.9790109190008228, 0.8988252193018174, 0.5867171662232342, 0.5881576704911717, 0.034267040352229494, 0.9985265777083543, 0.13157599736614178, 0.740347196631592, 0.8210151951243089, 0.3730545293052032, 0.19685205466531375, 0.09875988679055503, 0.7486060058295778, 0.4526535292056957, 0.7137177590011357, 0.9154076488518006, 0.1465837361510567, 0.9191710007237996, 0.4116264595084367, 0.30526700989728905, 0.9430622606027791, 0.9906516926063994, 0.19889221776744814, 0.6568383469519833, 0.10649531377106036, 0.6509140038575058, 0.8273132277758497, 0.6844985465240676, 0.41733314206259575, 0.38306635956376955, 0.39312241522341707, 0.5897118179929232, 0.8815672700724956, 0.9290661572687678, 0.05352962020731811, 0.1816223946456883, 0.11222431582828851, 0.19333464076691398, 0.3466078106091718, 0.5065316826226238, 0.6294612270091522, 0.7321422191397015, 0.8901115413858071, 0.9890884372297908, 0.6628564785571198, 0.845364518667763, 0.7780388469246989, 0.30753203921871197, 0.875692270234923, 0.042763137947798846, 0.00036734375145786036, 0.2737326293884642, 0.4620975296274499, 0.6383628950006773, 0.1017702665445448, 0.67301013383484, 0.8018158670596945, 0.1853129195793668, 0.41512525483179386, 0.5199849899533098, 0.45180701808448065, 0.7998299308884707, 0.9605223981193308, 0.79895316400818, 0.07799281787582457, 0.8049355721591271, 0.06659633223555117, 0.23597038524157, 0.1530968968357228, 0.19751910684608542, 0.5283151270573506, 0.6716898576108536, 0.470321282336041, 0.9596956390292436, 0.24029232003016665, 0.7631402302547929, 0.8701821785071485, 0.5620661107547191, 0.4562225019615531, 0.5961844467792436, 0.42880977016034294, 0.5551938823233694, 0.4169339521288016, 0.40046971029413403, 0.6953464681395298, 0.09285121306796795, 0.16654207272882082, 0.851198471715004, 0.771077346815188, 0.2814537272943938, 0.3772689326657602, 0.9260265066284805, 0.8180772251738613, 0.6143462999796309, 0.22149017880295774, 0.04425197131526959, 0.43125784684189816, 0.6726271392574619, 0.8284804905178516, 0.8526890057422694, 0.032775901293363385, 0.24415703383687937, 0.33909458847740304, 0.18873221095938586, 0.8029753783638378, 0.7674657630849425, 0.5168330403809708, 0.9829264779958555, 0.14405854109084226, 0.8996517033276689, 0.11646325416191616, 0.1631817055141207, 0.6962192002506188, 0.10956969204168931, 0.5658450954187109, 0.4202335364726536, 0.7284739655410148, 0.9006752388088651, 0.7698715146690553, 0.8496898764211883, 0.03294544853123238, 0.3101954982442031, 0.5154330866863324, 0.41595331462977947, 0.23125495297968846, 0.3078740979600233, 0.9454309703916057, 0.29418087987135155, 0.35390411017978574, 0.0037097737500186856, 0.8450776272778745, 0.15484070340157663, 0.20414427551372605, 0.2552645170920256, 0.8846220568448426, 0.20645141124631472, 0.7975263608679399, 0.8080493403229045, 0.9270205687863958, 0.11556131360031396, 0.21727897233783133, 0.7428982920807213, 0.19600075428073904, 0.286329546873449, 0.16674158011136264, 0.1726966861550493, 0.4815533546282057, 0.10968306211295853, 0.3216976184636222, 0.426593909598879, 0.024548116523977592, 0.3883331664744526, 0.09412243608551696, 0.49357853114617667, 0.8257381885570964, 0.8184221621817872, 0.08044851553740051, 0.6012277576745202, 0.8345863819712365, 0.23797254293463532, 0.7619265114495991, 0.8907643464500933, 0.8061241514404441, 0.10730103151550452, 0.009059999522384898, 0.19172410992184563, 0.27047734094662323, 0.6161829906159595, 0.3842731752423184, 0.7034070306614735, 0.3530749605216086, 0.1544254246546003, 0.3126898434860351, 0.884324226389135, 0.9585323442450648, 0.20751273406736848, 0.7884683870244413, 0.2733487365398707, 0.8871315434314075, 0.16554561279625546, 0.6659599186940569, 0.08421126471318252, 0.9738933239738695, 0.7006333446405428, 0.8418157394050853};
std::vector<float> fpweights_rand_pad {0.35798393686119634, 0.4351419865163296, 0.5909267255335692, 0.7223915186982364, 0.31763187327394615, 0.0, 0.0, 0.0, 0.3289537599189831, 0.019691642723703717, 0.040874860094018306, 0.2578216943085364, 0.7402449976749567, 0.0, 0.0, 0.0, 0.6283138303739122, 0.7697890206778347, 0.7689194362148217, 0.8565674676693013, 0.7203192659868836, 0.0, 0.0, 0.0, 0.9790109190008228, 0.8988252193018174, 0.5867171662232342, 0.5881576704911717, 0.034267040352229494, 0.0, 0.0, 0.0, 0.9985265777083543, 0.13157599736614178, 0.740347196631592, 0.8210151951243089, 0.3730545293052032, 0.0, 0.0, 0.0, 0.19685205466531375, 0.09875988679055503, 0.7486060058295778, 0.4526535292056957, 0.7137177590011357, 0.0, 0.0, 0.0, 0.9154076488518006, 0.1465837361510567, 0.9191710007237996, 0.4116264595084367, 0.30526700989728905, 0.0, 0.0, 0.0, 0.9430622606027791, 0.9906516926063994, 0.19889221776744814, 0.6568383469519833, 0.10649531377106036, 0.0, 0.0, 0.0, 0.6509140038575058, 0.8273132277758497, 0.6844985465240676, 0.41733314206259575, 0.38306635956376955, 0.0, 0.0, 0.0, 0.39312241522341707, 0.5897118179929232, 0.8815672700724956, 0.9290661572687678, 0.05352962020731811, 0.0, 0.0, 0.0, 0.1816223946456883, 0.11222431582828851, 0.19333464076691398, 0.3466078106091718, 0.5065316826226238, 0.0, 0.0, 0.0, 0.6294612270091522, 0.7321422191397015, 0.8901115413858071, 0.9890884372297908, 0.6628564785571198, 0.0, 0.0, 0.0, 0.845364518667763, 0.7780388469246989, 0.30753203921871197, 0.875692270234923, 0.042763137947798846, 0.0, 0.0, 0.0, 0.00036734375145786036, 0.2737326293884642, 0.4620975296274499, 0.6383628950006773, 0.1017702665445448, 0.0, 0.0, 0.0, 0.67301013383484, 0.8018158670596945, 0.1853129195793668, 0.41512525483179386, 0.5199849899533098, 0.0, 0.0, 0.0, 0.45180701808448065, 0.7998299308884707, 0.9605223981193308, 0.79895316400818, 0.07799281787582457, 0.0, 0.0, 0.0, 0.8049355721591271, 0.06659633223555117, 0.23597038524157, 0.1530968968357228, 0.19751910684608542, 0.0, 0.0, 0.0, 0.5283151270573506, 0.6716898576108536, 0.470321282336041, 0.9596956390292436, 0.24029232003016665, 0.0, 0.0, 0.0, 0.7631402302547929, 0.8701821785071485, 0.5620661107547191, 0.4562225019615531, 0.5961844467792436, 0.0, 0.0, 0.0, 0.42880977016034294, 0.5551938823233694, 0.4169339521288016, 0.40046971029413403, 0.6953464681395298, 0.0, 0.0, 0.0, 0.09285121306796795, 0.16654207272882082, 0.851198471715004, 0.771077346815188, 0.2814537272943938, 0.0, 0.0, 0.0, 0.3772689326657602, 0.9260265066284805, 0.8180772251738613, 0.6143462999796309, 0.22149017880295774, 0.0, 0.0, 0.0, 0.04425197131526959, 0.43125784684189816, 0.6726271392574619, 0.8284804905178516, 0.8526890057422694, 0.0, 0.0, 0.0, 0.032775901293363385, 0.24415703383687937, 0.33909458847740304, 0.18873221095938586, 0.8029753783638378, 0.0, 0.0, 0.0, 0.7674657630849425, 0.5168330403809708, 0.9829264779958555, 0.14405854109084226, 0.8996517033276689, 0.0, 0.0, 0.0, 0.11646325416191616, 0.1631817055141207, 0.6962192002506188, 0.10956969204168931, 0.5658450954187109, 0.0, 0.0, 0.0, 0.4202335364726536, 0.7284739655410148, 0.9006752388088651, 0.7698715146690553, 0.8496898764211883, 0.0, 0.0, 0.0, 0.03294544853123238, 0.3101954982442031, 0.5154330866863324, 0.41595331462977947, 0.23125495297968846, 0.0, 0.0, 0.0, 0.3078740979600233, 0.9454309703916057, 0.29418087987135155, 0.35390411017978574, 0.0037097737500186856, 0.0, 0.0, 0.0, 0.8450776272778745, 0.15484070340157663, 0.20414427551372605, 0.2552645170920256, 0.8846220568448426, 0.0, 0.0, 0.0, 0.20645141124631472, 0.7975263608679399, 0.8080493403229045, 0.9270205687863958, 0.11556131360031396, 0.0, 0.0, 0.0, 0.21727897233783133, 0.7428982920807213, 0.19600075428073904, 0.286329546873449, 0.16674158011136264, 0.0, 0.0, 0.0, 0.1726966861550493, 0.4815533546282057, 0.10968306211295853, 0.3216976184636222, 0.426593909598879, 0.0, 0.0, 0.0, 0.024548116523977592, 0.3883331664744526, 0.09412243608551696, 0.49357853114617667, 0.8257381885570964, 0.0, 0.0, 0.0, 0.8184221621817872, 0.08044851553740051, 0.6012277576745202, 0.8345863819712365, 0.23797254293463532, 0.0, 0.0, 0.0, 0.7619265114495991, 0.8907643464500933, 0.8061241514404441, 0.10730103151550452, 0.009059999522384898, 0.0, 0.0, 0.0, 0.19172410992184563, 0.27047734094662323, 0.6161829906159595, 0.3842731752423184, 0.7034070306614735, 0.0, 0.0, 0.0, 0.3530749605216086, 0.1544254246546003, 0.3126898434860351, 0.884324226389135, 0.9585323442450648, 0.0, 0.0, 0.0, 0.20751273406736848, 0.7884683870244413, 0.2733487365398707, 0.8871315434314075, 0.16554561279625546, 0.0, 0.0, 0.0, 0.6659599186940569, 0.08421126471318252, 0.9738933239738695, 0.7006333446405428, 0.8418157394050853, 0.0, 0.0, 0.0};
std::vector<float> fpbias_rand {0.5666693393630345, 0.4768013639288602, 0.6218823913943651, 0.528741511699448};

//BCHW
ConvReluGraphTest<ConvReluScalarBCHW, 28, 24, 1, 2, 4, 5> convReluScalarBCHW(
  "convReluScalarBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "conv_fpout_ConvReluScalarBCHW.txt");
ConvReluGraphTest<ConvReluScalarBCHW, 28, 24, 1, 2, 4, 5> convReluScalarBCHW_rand(
  "convReluScalarBCHW_rand", fpweights_rand, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarBCHW_rand.txt");

ConvReluGraphTest<Conv5x5ReluBCHW, 28, 24, 1, 2, 4, 5> conv5x5ReluBCHW(
  "conv5x5ReluBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "conv_fpout_Conv5x5ReluBCHW.txt");
ConvReluGraphTest<Conv5x5ReluBCHW, 28, 24, 1, 2, 4, 5> conv5x5ReluBCHW_rand(
  "conv5x5ReluBCHW_rand", fpweights_rand, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_Conv5x5ReluBCHW_rand.txt");

ConvReluGraphTest<Conv5x5on8ReluBCHW, 28, 24, 1, 2, 4, 5> conv5x5on8ReluBCHW(
  "conv5x5on8ReluBCHW", fpweights_pad, fpbias, 
  "conv_fpin.txt", "conv_fpout_Conv5x5on8ReluBCHW.txt");
ConvReluGraphTest<Conv5x5on8ReluBCHW, 28, 24, 1, 2, 4, 5> conv5x5on8ReluBCHW_rand(
  "conv5x5on8ReluBCHW_rand", fpweights_rand_pad, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_Conv5x5on8ReluBCHW_rand.txt");

//BHWC
ConvReluGraphTest<ConvReluScalarBHWC, 28, 24, 1, 2, 4, 5> convReluScalarBHWC(
  "convReluScalarBHWC", fpweights, fpbias, 
  "conv_fpin.txt", "conv_fpout_ConvReluScalarBHWC.txt");
ConvReluGraphTest<ConvReluScalarBHWC, 28, 24, 1, 2, 4, 5> convReluScalarBHWC_rand(
  "convReluScalarBHWC_rand", fpweights_rand, fpbias_rand, 
  "conv_fpin_rand.txt", "conv_fpout_ConvReluScalarBHWC_rand.txt");

ConvReluGmemParamGraphTest<ConvReluScalarGmemParamBHWC, 28, 24, 1, 2, 4, 5> convReluScalarGmemParamBHWC(
  "convReluScalarGmemParamBHWC", "conv_fpweights.txt", "conv_fpbias.txt", 
  "conv_fpin.txt", "conv_fpout_ConvReluScalarGmemParamBHWC.txt");
ConvReluGmemParamGraphTest<ConvReluScalarGmemParamBHWC, 28, 24, 1, 2, 4, 5> convReluScalarGmemParamBHWC_rand(
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
  get_graph_throughput_by_port(convReluScalarBCHW, "plout[0]", convReluScalarBCHW.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBCHW.end(), "end convReluScalarBCHW");

  adfCheck(convReluScalarBCHW_rand.init(), "init convReluScalarBCHW_rand");
  get_graph_throughput_by_port(convReluScalarBCHW_rand, "plout[0]", convReluScalarBCHW_rand.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBCHW_rand.end(), "end convReluScalarBCHW_rand");
  

  adfCheck(conv5x5ReluBCHW.init(), "init conv5x5ReluBCHW");
  get_graph_throughput_by_port(conv5x5ReluBCHW, "plout[0]", conv5x5ReluBCHW.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5ReluBCHW.end(), "end conv5x5ReluBCHW");

  adfCheck(conv5x5ReluBCHW_rand.init(), "init conv5x5ReluBCHW_rand");
  get_graph_throughput_by_port(conv5x5ReluBCHW_rand, "plout[0]", conv5x5ReluBCHW_rand.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5ReluBCHW_rand.end(), "end conv5x5ReluBCHW_rand");

  
  adfCheck(conv5x5on8ReluBCHW.init(), "init conv5x5on8ReluBCHW");
  get_graph_throughput_by_port(conv5x5on8ReluBCHW, "plout[0]", conv5x5on8ReluBCHW.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5on8ReluBCHW.end(), "end conv5x5on8ReluBCHW");

  adfCheck(conv5x5on8ReluBCHW_rand.init(), "init conv5x5on8ReluBCHW_rand");
  get_graph_throughput_by_port(conv5x5on8ReluBCHW_rand, "plout[0]", conv5x5on8ReluBCHW_rand.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(conv5x5on8ReluBCHW_rand.end(), "end conv5x5on8ReluBCHW_rand");


  // BHWC
  adfCheck(convReluScalarBHWC.init(), "init convReluScalarBHWC");
  get_graph_throughput_by_port(convReluScalarBHWC, "plout[0]", convReluScalarBHWC.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBHWC.end(), "end convReluScalarBHWC");

  adfCheck(convReluScalarBHWC_rand.init(), "init convReluScalarBHWC_rand");
  get_graph_throughput_by_port(convReluScalarBHWC_rand, "plout[0]", convReluScalarBHWC_rand.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarBHWC_rand.end(), "end convReluScalarBHWC_rand");


  adfCheck(convReluScalarGmemParamBHWC.init(), "init convReluScalarGmemParamBHWC");
  get_graph_throughput_by_port(convReluScalarGmemParamBHWC, "plout[0]", convReluScalarGmemParamBHWC.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarGmemParamBHWC.end(), "end convReluScalarGmemParamBHWC");

  adfCheck(convReluScalarGmemParamBHWC_rand.init(), "init convReluScalarGmemParamBHWC_rand");
  get_graph_throughput_by_port(convReluScalarGmemParamBHWC_rand, "plout[0]", convReluScalarGmemParamBHWC_rand.plout[0], 1*2*24*24, sizeof(float_t), ITER_CNT);
	adfCheck(convReluScalarGmemParamBHWC_rand.end(), "end convReluScalarGmemParamBHWC_rand");
  return 0;
}
#endif
