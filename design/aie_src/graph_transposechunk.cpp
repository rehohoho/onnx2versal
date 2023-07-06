#include "graph_transpose.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int> class TRANSPOSE, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int B, int H, int W, int C>
class TransposeHChunkGraphTest : public adf::graph {

  private:
    TransposeHChunkGraph<TRANSPOSE, CONCAT, HCHUNK, TT, B, H, W, C> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    TransposeHChunkGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "transpose_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_transpose"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_transpose"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<B*H*W*C*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*C*H*W*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
    }

};


// instance to be compiled and used in host within xclbin
TransposeHChunkGraphTest<TransposeScalarBHWC2BCHW, ConcatFloatStream, 2, float_t, 1, 4, 4, 16> transposeScalarBHWC2BCHW(
  "transposeScalarBHWC2BCHW", "transpose_fpin.txt", "transpose_fpout_shape1x16x4x4_TransposeScalarBHWC2BCHW.txt");


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(transposeScalarBHWC2BCHW.init(), "init transposeScalarBHWC2BCHW");
  adfCheck(transposeScalarBHWC2BCHW.run(ITER_CNT), "run transposeScalarBHWC2BCHW");
	adfCheck(transposeScalarBHWC2BCHW.end(), "end transposeScalarBHWC2BCHW");
  return 0;
}
#endif
