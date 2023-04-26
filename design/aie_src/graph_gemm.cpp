#include "graph_gemm.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
char gemm1_input[]  = "lenet_mnist__14__.fc1.Gemm__.Reshape_output_0__1x256.txt";
char gemm1_weight[] = "lenet_mnist__14__.fc1.Gemm__fc1.weight__120x256.txt";
char gemm1_bias[]   = "lenet_mnist__14__.fc1.Gemm__fc1.bias__120.txt";
char gemm1_output[] = "lenet_mnist__15__.relu3.Relu__.relu3.Relu_output_0__1x120.txt";
GemmReluScalar<1, 256, 120, 
  gemm1_input, gemm1_weight, gemm1_bias, gemm1_output> gemm1("1");

char gemm2_input[]  = "lenet_mnist__16__.fc2.Gemm__.relu3.Relu_output_0__1x120.txt";
char gemm2_weight[] = "lenet_mnist__16__.fc2.Gemm__fc2.weight__84x120.txt";
char gemm2_bias[]   = "lenet_mnist__16__.fc2.Gemm__fc2.bias__84.txt";
char gemm2_output[] = "lenet_mnist__17__.relu4.Relu__.relu4.Relu_output_0__1x84.txt";
GemmReluScalar<1, 120, 84, 
  gemm2_input, gemm2_weight, gemm2_bias, gemm2_output> gemm2("2");

char gemm3_input[]  = "lenet_mnist__18__.fc3.Gemm__.relu4.Relu_output_0__1x84.txt";
char gemm3_weight[] = "lenet_mnist__18__.fc3.Gemm__fc3.weight__10x84.txt";
char gemm3_bias[]   = "lenet_mnist__18__.fc3.Gemm__fc3.bias__10.txt";
char gemm3_output[] = "lenet_mnist__19__.relu5.Relu__output__1x10.txt";
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
