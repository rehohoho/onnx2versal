#include "graph_pool.h"
#include "graph_utils.h"

#define ITER_CNT 1


template <template<int, int, int, int> class POOL,
  int INP_W, int OUT_W, int B, int C>
class MaxpoolGraphTest : public adf::graph {

  private:
    MaxpoolGraph<POOL, INP_W, OUT_W, B, C> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    MaxpoolGraphTest(
      const std::string& id,
      const char* INP_TXT, 
      const char* OUT_TXT
    ) { 
      g.construct();
      plin[0] = adf::input_plio::create("plin0_maxpool"+id+"_input", adf::plio_64_bits, TXT_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_maxpool"+id+"_output", adf::plio_64_bits, TXT_ARG(OUT_TXT));
      adf::connect<adf::window<B*INP_W*INP_W*C*4>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*OUT_W*OUT_W*C*4>> (g.pout[0], plout[0].in[0]);
    }
};


// instance to be compiled and used in host within xclbin
MaxpoolGraphTest<MaxpoolScalarBHWC, 24, 12, 1, 6> scalar(
  "scalar", "pool_in.txt", "pool_out_MaxpoolScalarBHWC.txt");
MaxpoolGraphTest<MaxpoolScalarBHWC, 24, 12, 1, 6> scalar_rand(
  "scalar_rand", "pool_in_rand.txt", "pool_out_MaxpoolScalarBHWC_rand.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(scalar.init(), "init scalar");
  adfCheck(scalar.run(1), "run scalar");
	adfCheck(scalar.end(), "end scalar");

  adfCheck(scalar_rand.init(), "init scalar_rand");
  adfCheck(scalar_rand.run(1), "run scalar_rand");
	adfCheck(scalar_rand.end(), "end scalar_rand");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(scalar.init(), "init scalar");
  get_graph_throughput_by_port(scalar, "plout[0]", scalar.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(scalar.end(), "end scalar");

  adfCheck(scalar_rand.init(), "init scalar_rand");
  get_graph_throughput_by_port(scalar_rand, "plout[0]", scalar_rand.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(scalar_rand.end(), "end scalar_rand");
  return 0;
}
#endif
