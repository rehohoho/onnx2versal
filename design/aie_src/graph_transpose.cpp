#include "graph_transpose.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
char transpose1_input[]  = "lenet_mnist__13___Reshape___pool2_MaxPool_output_0__1x4x4x16.txt";
char transpose1_output[] = "lenet_mnist__13___Reshape___Reshape_output_0__1x256.txt";
TransposeScalar<1, 4, 4, 16, transpose1_input, transpose1_output> transpose1("1");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	
  adfCheck(transpose1.init(), "init transpose1");
  adfCheck(transpose1.run(1), "run transpose1");
	adfCheck(transpose1.end(), "end transpose1");

  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(transpose1.init(), "init transpose1");
  get_graph_throughput_by_port(transpose1, "plout[0]", transpose1.plout[0], 1*4*4*16, sizeof(float32), ITER_CNT);
	adfCheck(transpose1.end(), "end transpose1");

  return 0;
}
#endif
