#include "graph_identity.h"
#include "graph_utils.h"


template <typename TT, int N>
class IdentityGraphTest : public adf::graph {

  private:
    IdentityGraph<TT, N> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    IdentityGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "identity_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_iden"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_iden"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
IdentityGraphTest<float_t, 128> fpscalar(
  "fpscalar", "concat_fpin.txt", "concat_fpin.txt");

#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(fpscalar.init(), "init fpscalar");
  adfCheck(fpscalar.run(ITER_CNT), "run fpscalar");
	adfCheck(fpscalar.end(), "end fpscalar");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  adfCheck(fpscalar.init(), "init fpscalar");
  get_graph_throughput_by_port(fpscalar, "plout[0]", fpscalar.plout[0], 1*8, sizeof(float), ITER_CNT);
	adfCheck(fpscalar.end(), "end fpscalar");
  return 0;
}
#endif
