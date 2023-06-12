#include "graph_add.h"
#include "graph_utils.h"


template <template<typename, int, int> class ADD, 
  typename TT, int W, int IS_RELU>
class AddGraphTest : public adf::graph {

  private:
    static constexpr int TTSIZE = sizeof(TT);
    AddGraph<ADD, TT, W, IS_RELU> g;

  public:
    adf::input_plio plin[2];
    adf::output_plio plout[1];

    AddGraphTest(
      const std::string& id,
      const std::string& INPA_TXT, 
      const std::string& INPB_TXT, 
      const std::string& OUT_TXT
    ): g(8) { 
      plin[0] = adf::input_plio::create("plin0_add"+id, PLIO64_ARG(INPA_TXT));
      plin[1] = adf::input_plio::create("plin1_add"+id, PLIO64_ARG(INPB_TXT));
      plout[0] = adf::output_plio::create("plout0_add"+id, PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<W*TTSIZE>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<W*TTSIZE>> (plin[1].out[0], g.pin[1]);
      adf::connect<adf::window<W*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
AddGraphTest<AddScalar, float_t, 2048, 1> addScalar(
  "addScalar", 
  "k6add_inA_shape1x16x32x32.txt", 
  "k6add_inB_shape1x16x32x32.txt",
  "k6add_goldenout_shape1x16x32x32_scalar.txt");

AddGraphTest<AddFloat, float_t, 2048, 1> addFloat(
  "addFloat", 
  "k6add_inA_shape1x16x32x32.txt", 
  "k6add_inB_shape1x16x32x32.txt",
  "k6add_goldenout_shape1x16x32x32_float.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
	adfCheck(addScalar.init(), "init addScalar");
  adfCheck(addScalar.run(ITER_CNT), "run addScalar");
	adfCheck(addScalar.end(), "end addScalar");

  adfCheck(addFloat.init(), "init addFloat");
  adfCheck(addFloat.run(ITER_CNT), "run addFloat");
	adfCheck(addFloat.end(), "end addFloat");
  return 0;
}
#endif
