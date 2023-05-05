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
char pool1_input[]  = "lenet_mnist__2___pool1_MaxPool___relu1_Relu_output_0__1x24x24x6.txt";
char pool1_output[] = "lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x12x12x6.txt";
char pool2_input[]  = "lenet_mnist__5___pool2_MaxPool___relu2_Relu_output_0__1x8x8x16.txt";
char pool2_output[] = "lenet_mnist__5___pool2_MaxPool___pool2_MaxPool_output_0__1x4x4x16.txt";
MaxpoolGraphTest<MaxpoolScalarBHWC, 24, 12, 1, 6> pool1("1", pool1_input, pool1_output);
MaxpoolGraphTest<MaxpoolScalarBHWC, 8, 4, 1, 16> pool2("2", pool2_input, pool2_output);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(pool1.init(), "init pool1");
  adfCheck(pool1.run(1), "run pool1");
	adfCheck(pool1.end(), "end pool1");

  adfCheck(pool2.init(), "init pool2");
  adfCheck(pool2.run(1), "run pool2");
	adfCheck(pool2.end(), "end pool2");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(pool1.init(), "init pool1");
  get_graph_throughput_by_port(pool1, "plout[0]", pool1.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(pool1.end(), "end pool1");

  adfCheck(pool2.init(), "init pool2");
  get_graph_throughput_by_port(pool2, "plout[0]", pool2.plout[0], 1*4*4*16, sizeof(float32), ITER_CNT);
	adfCheck(pool2.end(), "end pool2");

  return 0;
}
#endif
