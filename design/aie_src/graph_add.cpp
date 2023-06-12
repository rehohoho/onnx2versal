#include "graph_add.h"
#include "graph_utils.h"


template <template<typename, int, int, int> class ADD, 
  typename TT, int B, int W, int IS_RELU>
class AddGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    AddGraph<ADD, TT, B, W, IS_RELU> g;

  public:
    adf::input_plio plin[2];
    adf::output_plio plout[1];

    AddGraphTest(
      const std::string& id,
      const std::string& INPA_TXT, 
      const std::string& INPB_TXT, 
      const std::string& OUT_TXT
    ) { 
      plin[0] = adf::input_plio::create("plin0_add"+id, PLIO64_ARG(INPA_TXT));
      plin[1] = adf::input_plio::create("plin1_add"+id, PLIO64_ARG(INPB_TXT));
      plout[0] = adf::output_plio::create("plout0_add"+id, PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*W*TTSIZE>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*W*TTSIZE>> (plin[1].out[0], g.pin[1]);
      adf::connect<adf::window<B*W*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
// padded to vector boundary
AddGraphTest<AddScalar, float_t, 1, 16384, 1> addScalar(
  "addScalar", 
  "k6add_inA_shape1x16x32x32.txt", 
  "k6add_inB_shape1x16x32x32.txt",
  "k6add_goldenout_shape1x16x32x32.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(addScalar.init(), "init addScalar");
  adfCheck(addScalar.run(ITER_CNT), "run addScalar");
	adfCheck(addScalar.end(), "end addScalar");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(addScalar.init(), "init addScalar");
  get_graph_throughput_by_port(addScalar, "plout[0]", addScalar.plout[0], 10, sizeof(float), ITER_CNT);
	adfCheck(addScalar.end(), "end addScalar");
  return 0;
}
#endif
