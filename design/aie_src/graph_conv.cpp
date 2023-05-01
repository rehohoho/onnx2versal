#include "graph_conv.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
std::vector<float> test_weights({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
std::vector<float> test_bias({1, 1, 1, 1, 1, 1});
ConvReluScalarBchwGraph<28, 24, 1, 1, 6, 5> conv_bchw(
  "bchw", test_weights, test_bias, "conv_in.txt", "conv_bchw_out.txt");
ConvReluScalarBhwcGraph<28, 24, 1, 1, 6, 5> conv_bhwc(
  "bhwc", test_weights, test_bias, "conv_in.txt", "conv_bhwc_out.txt");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
  adfCheck(conv_bchw.init(), "init conv_bchw");
  adfCheck(conv_bchw.run(1), "run conv_bchw");
	adfCheck(conv_bchw.end(), "end conv_bchw");

  adfCheck(conv_bhwc.init(), "init conv_bhwc");
  adfCheck(conv_bhwc.run(1), "run conv_bhwc");
	adfCheck(conv_bhwc.end(), "end conv_bhwc");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(conv_bchw.init(), "init conv_bchw");
  get_graph_throughput_by_port(conv_bchw, "plout[0]", conv_bchw.plout[0], 1*24*24*6, sizeof(float_t), ITER_CNT);
	adfCheck(conv_bchw.end(), "end conv_bchw");

  adfCheck(conv_bhwc.init(), "init conv_bhwc");
  get_graph_throughput_by_port(conv_bhwc, "plout[0]", conv_bhwc.plout[0], 1*24*24*6, sizeof(float_t), ITER_CNT);
	adfCheck(conv_bhwc.end(), "end conv_bhwc");

  return 0;
}
#endif
