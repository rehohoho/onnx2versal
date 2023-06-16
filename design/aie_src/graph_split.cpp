#include "graph_split.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class SPLIT,
  typename TT, int H, int INP_W, int OUT_W, int OVERLAP = 0>
class SplitGraphTest : public adf::graph {

  private:
    typedef SplitGraph<SPLIT, TT, H, INP_W, OUT_W, OVERLAP> Graph;
    Graph g;
    static constexpr int LCNT = Graph::LCNT;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[LCNT];

    SplitGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::vector<std::string> OUT_TXT = std::vector<std::string>()
    ) {
      plin[0] = adf::input_plio::create("plin0_split_"+id+"_input", PLIO64_ARG(INP_TXT));      
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);

      for (int i = 0; i < LCNT; i++) {
        plout[i] = adf::output_plio::create("plout"+std::to_string(i)+"_split_"+id+"_input", PLIO64_ARG(OUT_TXT[i]));
        adf::connect<adf::window<H*OUT_W*sizeof(TT)>> (g.pout[i], plout[i].in[0]);
      }
    }

};


// instance to be compiled and used in host within xclbin
const int H = 10;
const int INP_W = 64;
const int OUT_W = 22;
const int OVERLAP = 1;

SplitGraphTest<SplitScalar, float_t, H, INP_W, OUT_W, OVERLAP> splitScalar(
  "splitScalar", "split_fpin.txt", 
  {
    "split_fpout0_shape10x22_splitScalar.txt", 
    "split_fpout1_shape10x22_splitScalar.txt",
    "split_fpout2_shape10x22_splitScalar.txt"
  });

SplitGraphTest<SplitScalar, float_t, H, INP_W, 31, -1> splitScalar_neg(
  "splitScalar_neg", "split_fpin.txt", 
  {
    "split_fpout0_shape10x31_splitScalar.txt", 
    "split_fpout1_shape10x31_splitScalar.txt"
  });

#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(splitScalar.init(), "init splitScalar");
  adfCheck(splitScalar.run(ITER_CNT), "run splitScalar");
	adfCheck(splitScalar.end(), "end splitScalar");

  adfCheck(splitScalar_neg.init(), "init splitScalar_neg");
  adfCheck(splitScalar_neg.run(ITER_CNT), "run splitScalar_neg");
	adfCheck(splitScalar_neg.end(), "end splitScalar_neg");
  return 0;
}
#endif
