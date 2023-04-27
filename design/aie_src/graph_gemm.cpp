#include "graph_gemm.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
char gemm1_input[]  = "lenet_mnist__14___fc1_Gemm___Reshape_output_0__1x256.txt";
char gemm1_weight[] = "lenet_mnist__14___fc1_Gemm__fc1_weight__120x256.txt";
char gemm1_bias[]   = "lenet_mnist__14___fc1_Gemm__fc1_bias__120.txt";
char gemm1_output[] = "lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt";
GemmReluScalar<1, 256, 120, 
  gemm1_input, gemm1_weight, gemm1_bias, gemm1_output> gemm1("1");

char gemm2_input[]  = "lenet_mnist__16___fc2_Gemm___relu3_Relu_output_0__1x120.txt";
char gemm2_weight[] = "lenet_mnist__16___fc2_Gemm__fc2_weight__84x120.txt";
char gemm2_bias[]   = "lenet_mnist__16___fc2_Gemm__fc2_bias__84.txt";
char gemm2_output[] = "lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt";
GemmReluScalar<1, 120, 84, 
  gemm2_input, gemm2_weight, gemm2_bias, gemm2_output> gemm2("2");

char gemm3_input[]  = "lenet_mnist__18___fc3_Gemm___relu4_Relu_output_0__1x84.txt";
char gemm3_weight[] = "lenet_mnist__18___fc3_Gemm__fc3_weight__10x84.txt";
char gemm3_bias[]   = "lenet_mnist__18___fc3_Gemm__fc3_bias__10.txt";
char gemm3_output[] = "lenet_mnist__19___relu5_Relu__output__1x10.txt";
GemmReluScalar<1, 84, 10, 
  gemm3_input, gemm3_weight, gemm3_bias, gemm3_output> gemm3("3");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(gemm1.init(), "init gemm1");
  adfCheck(gemm1.run(1), "run gemm1");
	adfCheck(gemm1.end(), "end gemm1");

  adfCheck(gemm2.init(), "init gemm2");
  adfCheck(gemm2.run(1), "run gemm2");
	adfCheck(gemm2.end(), "end gemm2");
  
  adfCheck(gemm3.init(), "init gemm3");
  adfCheck(gemm3.run(1), "run gemm3");
	adfCheck(gemm3.end(), "end gemm3");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	adfCheck(gemm1.init(), "init gemm1");
  get_graph_throughput_by_port(gemm1, "plout[0]", gemm1.plout[0], 1*120, sizeof(float32), ITER_CNT);
	adfCheck(gemm1.end(), "end gemm1");

  adfCheck(gemm2.init(), "init gemm2");
  get_graph_throughput_by_port(gemm2, "plout[0]", gemm2.plout[0], 1*84, sizeof(float32), ITER_CNT);
	adfCheck(gemm2.end(), "end gemm2");

  adfCheck(gemm3.init(), "init gemm3");
  get_graph_throughput_by_port(gemm3, "plout[0]", gemm3.plout[0], 1*10, sizeof(float32), ITER_CNT);
	adfCheck(gemm3.end(), "end gemm3");
  return 0;
}
#endif
