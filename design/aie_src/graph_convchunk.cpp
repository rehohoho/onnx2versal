#include "graph_conv.h"
#include "graph_utils.h"


template <
  template<int, int, int, int, int, int, int, int> class CONV, 
  template<typename, int, int, int, int> class CONCAT, 
  int IS_BCHW, int IS_KPAD, int MCHUNK, 
  int INP_H, int INP_W, int OUT_W, int B, int C, int M, int K, int IS_RELU,
  int H0 = 0, int H1 = 0, int W0 = 0, int W1 = 0>
class ConvReluChunkGraphTest : public adf::graph {

  private:
    typedef ConvReluChunkGraph<CONV, CONCAT, IS_BCHW, IS_KPAD, MCHUNK, 
                               INP_H, INP_W, OUT_W, B, C, M, K, IS_RELU,
                               H0, H1, W0, W1> Graph;
    Graph g;
    static constexpr int OUT_H = Graph::OUT_H;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    ConvReluChunkGraphTest(
      const std::string& id,
      std::vector<float> weights,
      std::vector<float> bias,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "conv_out.txt"
    ): g(weights, bias) { 
      plin[0] = adf::input_plio::create("plin0_conv"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_conv"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*C*INP_H*INP_W*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*M*OUT_H*OUT_W*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
const int INP_H = 24;
const int INP_W = 24;
const int OUT_W = INP_W;
const int B = 1;
const int C = 1; // loop dependency missed issue occurs at C=1
const int M = 5;
const int K = 5;
const int IS_RELU = 1;
const int MCHUNK = 2; // aiecompiler have issues when this is 1 (deadlock?) or 3 (invalid seg)
const int PAD = K/2;

std::vector<float> fpweights {0.16046764388760104, 0.8863046660865599, 0.4463944154832029, 0.9078755943543261, 0.16023046632014326, 0.6611175115080995, 0.4402637528294918, 0.0764867690302854, 0.6964631446525006, 0.2473987555391537, 0.039615522579517726, 0.05994429824957326, 0.06107853706678734, 0.9077329574850395, 0.739883917829101, 0.8980623572137351, 0.6725823112965214, 0.5289399290308832, 0.30444636434737826, 0.9979622513286734, 0.36218905893938935, 0.47064894921390954, 0.37824517492346177, 0.9795269293354586, 0.17465838539500578, 0.32798800090807967, 0.6803486660150015, 0.06320761833863064, 0.607249374011541, 0.4776465028764161, 0.2839999767621011, 0.238413280924058, 0.5145127432987567, 0.36792758053704133, 0.45651989126265535, 0.3374773817642399, 0.9704936935959776, 0.13343943174560402, 0.0968039531783742, 0.34339172879091606, 0.5910269008704913, 0.6591764718500283, 0.39725674716804205, 0.9992779939221711, 0.3518929961930426, 0.7214066679599525, 0.6375826945307929, 0.8130538632474607, 0.97622566345382, 0.8897936564455402, 0.7645619743577086, 0.6982484778182906, 0.3354981696758996, 0.14768557820670736, 0.06263600305980976, 0.24190170420148482, 0.4322814811812986, 0.5219962736299825, 0.7730835540548716, 0.958740923056593, 0.11732048038481102, 0.10700414019379156, 0.5896947230135507, 0.745398073947293, 0.8481503803469849, 0.9358320802167885, 0.983426242260642, 0.3998016922245259, 0.3803351835275731, 0.14780867669727238, 0.6849344386835594, 0.6567619584408371, 0.8620625958512073, 0.09725799478764063, 0.4977769078253418, 0.5810819296720631, 0.2415570400399184, 0.16902540612916128, 0.8595808364196215, 0.058534922235558784, 0.4706209039180729, 0.11583400130088528, 0.45705876133136736, 0.9799623263423093, 0.4237063534554728, 0.8571249175045673, 0.11731556418319389, 0.27125207676186414, 0.4037927406673345, 0.39981214000933074, 0.6713834786701531, 0.34471812737550767, 0.7137668684100164, 0.6391868992253925, 0.3991611452547731, 0.43176012765431926, 0.6145276998103207, 0.07004219014464463, 0.8224067383556903, 0.6534211611136369, 0.7263424644178352, 0.5369230010823904, 0.11047711099174473, 0.4050356132969499, 0.40537358284855607, 0.3210429900432169, 0.02995032490474936, 0.7372542425964773, 0.1097844580625007, 0.6063081330450851, 0.7032174964672158, 0.6347863229336947, 0.959142251977975, 0.10329815508513862, 0.8671671591051991, 0.029190234848913255, 0.534916854927084, 0.4042436179392588, 0.5241838603937582, 0.3650998770600098, 0.19056691494006806, 0.01912289744868978, 0.5181498137911743, 0.8427768626848423, 0.37321595574479416};
std::vector<float> fpweights_pad {0.16046764388760104, 0.8863046660865599, 0.4463944154832029, 0.9078755943543261, 0.16023046632014326, 0.0, 0.0, 0.0, 0.6611175115080995, 0.4402637528294918, 0.0764867690302854, 0.6964631446525006, 0.2473987555391537, 0.0, 0.0, 0.0, 0.039615522579517726, 0.05994429824957326, 0.06107853706678734, 0.9077329574850395, 0.739883917829101, 0.0, 0.0, 0.0, 0.8980623572137351, 0.6725823112965214, 0.5289399290308832, 0.30444636434737826, 0.9979622513286734, 0.0, 0.0, 0.0, 0.36218905893938935, 0.47064894921390954, 0.37824517492346177, 0.9795269293354586, 0.17465838539500578, 0.0, 0.0, 0.0, 0.32798800090807967, 0.6803486660150015, 0.06320761833863064, 0.607249374011541, 0.4776465028764161, 0.0, 0.0, 0.0, 0.2839999767621011, 0.238413280924058, 0.5145127432987567, 0.36792758053704133, 0.45651989126265535, 0.0, 0.0, 0.0, 0.3374773817642399, 0.9704936935959776, 0.13343943174560402, 0.0968039531783742, 0.34339172879091606, 0.0, 0.0, 0.0, 0.5910269008704913, 0.6591764718500283, 0.39725674716804205, 0.9992779939221711, 0.3518929961930426, 0.0, 0.0, 0.0, 0.7214066679599525, 0.6375826945307929, 0.8130538632474607, 0.97622566345382, 0.8897936564455402, 0.0, 0.0, 0.0, 0.7645619743577086, 0.6982484778182906, 0.3354981696758996, 0.14768557820670736, 0.06263600305980976, 0.0, 0.0, 0.0, 0.24190170420148482, 0.4322814811812986, 0.5219962736299825, 0.7730835540548716, 0.958740923056593, 0.0, 0.0, 0.0, 0.11732048038481102, 0.10700414019379156, 0.5896947230135507, 0.745398073947293, 0.8481503803469849, 0.0, 0.0, 0.0, 0.9358320802167885, 0.983426242260642, 0.3998016922245259, 0.3803351835275731, 0.14780867669727238, 0.0, 0.0, 0.0, 0.6849344386835594, 0.6567619584408371, 0.8620625958512073, 0.09725799478764063, 0.4977769078253418, 0.0, 0.0, 0.0, 0.5810819296720631, 0.2415570400399184, 0.16902540612916128, 0.8595808364196215, 0.058534922235558784, 0.0, 0.0, 0.0, 0.4706209039180729, 0.11583400130088528, 0.45705876133136736, 0.9799623263423093, 0.4237063534554728, 0.0, 0.0, 0.0, 0.8571249175045673, 0.11731556418319389, 0.27125207676186414, 0.4037927406673345, 0.39981214000933074, 0.0, 0.0, 0.0, 0.6713834786701531, 0.34471812737550767, 0.7137668684100164, 0.6391868992253925, 0.3991611452547731, 0.0, 0.0, 0.0, 0.43176012765431926, 0.6145276998103207, 0.07004219014464463, 0.8224067383556903, 0.6534211611136369, 0.0, 0.0, 0.0, 0.7263424644178352, 0.5369230010823904, 0.11047711099174473, 0.4050356132969499, 0.40537358284855607, 0.0, 0.0, 0.0, 0.3210429900432169, 0.02995032490474936, 0.7372542425964773, 0.1097844580625007, 0.6063081330450851, 0.0, 0.0, 0.0, 0.7032174964672158, 0.6347863229336947, 0.959142251977975, 0.10329815508513862, 0.8671671591051991, 0.0, 0.0, 0.0, 0.029190234848913255, 0.534916854927084, 0.4042436179392588, 0.5241838603937582, 0.3650998770600098, 0.0, 0.0, 0.0, 0.19056691494006806, 0.01912289744868978, 0.5181498137911743, 0.8427768626848423, 0.37321595574479416, 0.0, 0.0, 0.0};
std::vector<float> fpbias {0.22286381801498023, 0.08053200347184408, 0.08531092311870336, 0.22139644629277222, 0.10001406092155518};

// BCHW
ConvReluChunkGraphTest<ConvReluScalarBCHW, ConcatFloat, 1, 0, MCHUNK, 
                       INP_H, INP_W, OUT_W, B, C, M, K, IS_RELU,
                       PAD, PAD, PAD, PAD> convReluScalarBCHW(
  "convReluScalarBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x5x24x24_ConvReluScalarBCHW.txt");

ConvReluChunkGraphTest<Conv5x5ReluBCHW, ConcatFloat, 1, 0, MCHUNK, 
                       INP_H, INP_W, OUT_W, B, C, M, K, IS_RELU,
                       PAD, PAD, PAD, PAD> conv5x5ReluBCHW(
  "conv5x5ReluBCHW", fpweights, fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x5x24x24_Conv5x5ReluBCHW.txt");

ConvReluChunkGraphTest<Conv5x5on8ReluBCHW, ConcatFloat, 1, 1, MCHUNK, 
                       INP_H, INP_W, OUT_W, B, C, M, K, IS_RELU,
                       PAD, PAD, PAD, PAD> conv5x5on8ReluBCHW(
  "conv5x5on8ReluBCHW", fpweights_pad, fpbias, 
  "conv_fpin.txt", "convbchw_fpout_shape1x5x24x24_Conv5x5on8ReluBCHW.txt");

// BHWC, ConcatFloat requires CONCAT_BLOCK=M%4=0
ConvReluChunkGraphTest<ConvReluScalarBHWC, ConcatScalar, 0, 0, MCHUNK, 
                       INP_H, INP_W, OUT_W, B, C, M, K, IS_RELU,
                       PAD, PAD, PAD, PAD> convReluScalarBHWC(
  "convReluScalarBHWC", fpweights, fpbias, 
  "conv_fpin.txt", "convbhwc_fpout_shape1x5x24x24_ConvReluScalarBHWC.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
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

  // BHWC, ConcatFloat requires CONCAT_BLOCK=M%4=0
  adfCheck(convReluScalarBHWC.init(), "init convReluScalarBHWC");
  adfCheck(convReluScalarBHWC.run(ITER_CNT), "run convReluScalarBHWC");
	adfCheck(convReluScalarBHWC.end(), "end convReluScalarBHWC");
  return 0;
}
#endif
