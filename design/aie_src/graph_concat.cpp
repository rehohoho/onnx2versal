#include "graph_concat.h"
#include "graph_utils.h"


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatGraphTest : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
    ConcatGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatGraphTest(
      const std::string& id,
      const std::string& OUT_TXT = "concat_out.txt",
      const std::vector<std::string> INP_TXT = std::vector<std::string>()
    ) {
      for (int i = 0; i < LCNT; i++) {
        std::string plin_name = "plin"+std::to_string(i)+"_concat_"+id+"_input"; 
        plin[i] = adf::input_plio::create(plin_name, PLIO64_ARG(INP_TXT[i]));
        adf::connect<adf::window<H*INP_W*TTSIZE>> (plin[i].out[0], g.pin[i]);
      }

      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<H*OUT_W*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }

};


template <template<typename, int, int, int, int> class CONCAT,
  typename TT, int LCNT, int H, int INP_W, int OUT_W>
class ConcatTwiceGraphTest : public adf::graph {

  private:
    static const int TTSIZE = sizeof(TT);
    ConcatTwiceGraph<CONCAT, TT, LCNT, H, INP_W, OUT_W> g;

  public:
    adf::input_plio plin[LCNT];
    adf::output_plio plout[1];

    ConcatTwiceGraphTest(
      const std::string& id,
      const std::string& OUT_TXT = "concat_out.txt",
      const std::vector<std::string> INP_TXT = std::vector<std::string>()
    ) {
      for (int i = 0; i < LCNT; i++) {
        std::string plin_name = "plin"+std::to_string(i)+"_concat_"+id+"_input"; 
        plin[i] = adf::input_plio::create(plin_name, PLIO64_ARG(INP_TXT[i]));
        adf::connect<adf::window<H*INP_W*TTSIZE>> (plin[i].out[0], g.pin[i]);
      }
      
      plout[0] = adf::output_plio::create("plout0_concat_"+id+"_output", PLIO64_ARG(OUT_TXT));
      adf::connect<adf::window<H*OUT_W*TTSIZE>> (g.pout[0], plout[0].in[0]);
    }

};


const int LCNT = 5; 
const int CHUNK_CNT = 4;
const int CHUNK_SIZE = 32; // % 16 for int8, % 8 for float
const int BLOCK_SIZE = 112; //% 16 for int8, % 4 for float

std::vector<std::string> fp_txts {
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt"
};

// instance to be compiled and used in host within xclbin
ConcatGraphTest<ConcatScalar, float_t, LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE> concatScalar(
  "concatScalar", "concat_fpout_shape4x112_ConcatScalar.txt", fp_txts);

ConcatGraphTest<ConcatFloat, float_t, LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE> concatFloat(
  "concatFloat", "concat_fpout_shape4x112_ConcatFloat.txt", fp_txts);

std::vector<std::string> int8_txts {
  "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", "concat_int8in.txt", 
  "concat_int8in.txt"
};

ConcatGraphTest<ConcatInt8, int8_t, LCNT, CHUNK_CNT, CHUNK_SIZE, BLOCK_SIZE> concatInt8(
  "concatInt8", "concat_int8out_shape4x112_ConcatInt8.txt", int8_txts);

const int MULTI_LCNT = 52; 
const int MULTI_BLOCK_SIZE = 816; //% 16 for int8, % 4 for float

std::vector<std::string> multi_fp_txts {
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", 
  "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt", "concat_fpin.txt"
};

// ConcatTwiceGraphTest<ConcatFloat, float_t, MULTI_LCNT, CHUNK_CNT, CHUNK_SIZE, MULTI_BLOCK_SIZE> multiConcatFloat(
//   "multiConcatFloat", "concatmulti_fpout_shape4x816_ConcatFloat.txt", multi_fp_txts
// );


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {
  adfCheck(concatScalar.init(), "init concatScalar");
  adfCheck(concatScalar.run(ITER_CNT), "run concatScalar");
	adfCheck(concatScalar.end(), "end concatScalar");

  adfCheck(concatFloat.init(), "init concatFloat");
  adfCheck(concatFloat.run(ITER_CNT), "run concatFloat");
	adfCheck(concatFloat.end(), "end concatFloat");

  adfCheck(concatInt8.init(), "init concatInt8");
  adfCheck(concatInt8.run(ITER_CNT), "run concatInt8");
	adfCheck(concatInt8.end(), "end concatInt8");

  // not very feasible mapping into array
  // adfCheck(multiConcatFloat.init(), "init multiConcatFloat");
  // adfCheck(multiConcatFloat.run(ITER_CNT), "run multiConcatFloat");
	// adfCheck(multiConcatFloat.end(), "end multiConcatFloat");
  return 0;
}
#endif
