#include "graph_concat.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
ConcatScalarGraph<5, 8, 8, 4*8+4> concat1("1",
  "concat_in.txt", // INP0_TXT
  "concat_in.txt", // INP1_TXT
  "concat_in.txt", // INP2_TXT
  "concat_in.txt", // INP3_TXT
  "concat_in.txt", // INP4_TXT
  "empty.txt",     // INP5_TXT
  "empty.txt",     // INP6_TXT
  "empty.txt",     // INP7_TXT
  "concat1_out.txt"// OUT_TXT
);

ConcatScalarGraph<5, 8, 4, 4*4+2> concat2("2",
  "concat_in.txt", // INP0_TXT
  "concat_in.txt", // INP1_TXT
  "concat_in.txt", // INP2_TXT
  "concat_in.txt", // INP3_TXT
  "concat_in.txt", // INP4_TXT
  "empty.txt",     // INP5_TXT
  "empty.txt",     // INP6_TXT
  "empty.txt",     // INP7_TXT
  "concat2_out.txt"// OUT_TXT
);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	
  adfCheck(concat1.init(), "init concat1");
  adfCheck(concat1.run(1), "run concat1");
	adfCheck(concat1.end(), "end concat1");

  adfCheck(concat2.init(), "init concat2");
  adfCheck(concat2.run(1), "run concat2");
	adfCheck(concat2.end(), "end concat2");

  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(concat1.init(), "init concat1");
  get_graph_throughput_by_port(concat1, "plout[0]", concat1.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(concat1.end(), "end concat1");

  adfCheck(concat2.init(), "init concat2");
  get_graph_throughput_by_port(concat2, "plout[0]", concat2.plout[0], 36, sizeof(float), ITER_CNT);
	adfCheck(concat2.end(), "end concat2");

  return 0;
}
#endif
