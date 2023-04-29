#include "graph_concat.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
ConcatScalarGraph<3, 8, 8, 8, 0, 18> concat1("1",
  "concat_in.txt", // INP0_TXT
  "concat_in.txt", // INP1_TXT
  "concat_in.txt", // INP2_TXT
  "empty.txt",     // INP3_TXT
  "concat_out.txt" // OUT_TXT
);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	
  adfCheck(concat1.init(), "init concat1");
  adfCheck(concat1.run(1), "run concat1");
	adfCheck(concat1.end(), "end concat1");

  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(concat1.init(), "init concat1");
  get_graph_throughput_by_port(concat1, "plout[0]", concat1.plout[0], 8+8+8+8, sizeof(float), ITER_CNT);
	adfCheck(concat1.end(), "end concat1");

  return 0;
}
#endif
