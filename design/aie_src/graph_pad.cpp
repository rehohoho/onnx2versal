#include "graph_pad.h"
#include "graph_utils.h"


template <template<typename, int, int, int> class PAD, 
  typename TT, int N, int INP_W, int OUT_W>
class PadGraphTest : public adf::graph {

  private:
    PadGraph<PAD, TT, N, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    PadGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "pad_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_pad_"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_pad_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<N*INP_W*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<N*OUT_W*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
PadGraphTest<PadScalar, int16_t, 28, 28, 32> padScalar(
  "padScalar", "pad_int16in.txt", "pad_int16out_PadScalar.txt");
PadGraphTest<PadVectorInt16, int16_t, 28, 28, 32> padVector(
  "padVector", "pad_int16in.txt", "pad_int16out_PadVector.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(padScalar.init(), "init padScalar");
  adfCheck(padScalar.run(ITER_CNT), "run padScalar");
	adfCheck(padScalar.end(), "end padScalar");

  adfCheck(padVector.init(), "init padVector");
  adfCheck(padVector.run(ITER_CNT), "run padVector");
	adfCheck(padVector.end(), "end padVector");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
  adfCheck(padScalar.init(), "init padScalar");
  get_graph_throughput_by_port(padScalar, "plout[0]", padScalar.plout[0], 28*32, sizeof(int16_t), ITER_CNT);
	adfCheck(padScalar.end(), "end padScalar");

  adfCheck(padVector.init(), "init padVector");
  get_graph_throughput_by_port(padVector, "plout[0]", padVector.plout[0], 28*32, sizeof(int16_t), ITER_CNT);
	adfCheck(padVector.end(), "end padVector");
  return 0;
}
#endif
