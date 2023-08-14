#include "graph_transpose.h"
#include "graph_utils.h"


template <
  template<typename, int, int, int, int, int> class TRANSPOSE, 
  int HCHUNK,
  typename TT, int B, int H, int W, int C, int PAD_W>
class TransposeHChunkGraphTest : public adf::graph {

  private:
    TransposeHChunkGraph<TRANSPOSE, HCHUNK, TT, B, H, W, C, PAD_W> g;

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
      adf::connect<adf::window<B*H*PAD_W*C*sizeof(TT)>> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::stream> (g.pout[0], plout[0].in[0]);
    }

};


template <
  template<typename, int, int, int, int> class SPLIT,
  template<typename, int, int, int, int, int> class TRANSPOSE, 
  template<typename, int, int, int, int> class CONCAT, 
  int HCHUNK,
  typename TT, int B, int H, int W, int C, int PAD_W>
class TransposeChunkHPktStreamGraphTest : public adf::graph {

  private:
    TransposeChunkHPktStreamGraph<SPLIT, TRANSPOSE, CONCAT, HCHUNK, TT, B, H, W, C, PAD_W> g;

  public:
    adf::input_plio plin[1];
    adf::output_plio plout[1];

    TransposeChunkHPktStreamGraphTest(
      const std::string& id,
      const std::string& INP_TXT,
      const std::string& OUT_TXT = "transpose_out.txt"
    ) { 
      plin[0] = adf::input_plio::create("plin0_transpose"+id+"_input", PLIO64_ARG(INP_TXT));
      plout[0] = adf::output_plio::create("plout0_transpose"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::stream> (plin[0].out[0], g.pin[0]);
      adf::connect<adf::window<B*C*H*W*sizeof(TT)>> (g.pout[0], plout[0].in[0]);
    }

};


TransposeChunkHPktStreamGraphTest<SplitFilterFloatPktStream, TransposeScalarPktStreamBHWC2BCHW, ConcatFloatStream, 2, 
                                  float_t, 1, 4, 4, 16, 4> transposeScalarBHWC2BCHW(
  "transposeScalarBHWC2BCHW", "transpose_fp_bchw_shape1x4x4x16.txt", "transpose_fp_bhwc_shape1x16x4x4_TransposeScalarBHWC2BCHW.txt");

TransposeHChunkGraphTest<TransposeScalarBHWC2BCHWStream,24,float,1,96,96,3,96> k000transpose(
  "k000transpose", 
  "k000transpose_in_shape1x96x96x3.txt",
  "k000transpose_goldenout_shape1x3x96x96.txt"
);

#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(transposeScalarBHWC2BCHW.init(), "init transposeScalarBHWC2BCHW");
  adfCheck(transposeScalarBHWC2BCHW.run(ITER_CNT), "run transposeScalarBHWC2BCHW");
	adfCheck(transposeScalarBHWC2BCHW.end(), "end transposeScalarBHWC2BCHW");

  adfCheck(k000transpose.init(), "init k000transpose");
  adfCheck(k000transpose.run(ITER_CNT), "run k000transpose");
	adfCheck(k000transpose.end(), "end k000transpose");
  return 0;
}
#endif
