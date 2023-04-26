#include "graph_conv.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
// illegal shit here
char conv1_input[]  = "lenet_mnist__0__.conv1.Conv__input__1x28x28x1.txt";
char conv1_weight[] = "lenet_mnist__0__.conv1.Conv__conv1.weight__6x5x5x1.txt";
char conv1_bias[]   = "lenet_mnist__0__.conv1.Conv__conv1.bias__6.txt";
char conv1_output[] = "lenet_mnist__0__.conv1.Conv__.conv1.Conv_output_0__1x24x24x6.txt";
ConvScalar<28, 24, 1, 1, 6, 5, 
           conv1_input, conv1_weight, conv1_bias, conv1_output> conv1("1");

char conv3_input[]  = "lenet_mnist__3__.conv2.Conv__.pool1.MaxPool_output_0__1x12x12x6.txt";
char conv3_weight[] = "lenet_mnist__3__.conv2.Conv__conv2.weight__16x5x5x6.txt";
char conv3_bias[]   = "lenet_mnist__3__.conv2.Conv__conv2.bias__16.txt";
char conv3_output[] = "lenet_mnist__3__.conv2.Conv__.conv2.Conv_output_0__1x8x8x16.txt";
ConvScalar<12, 8, 1, 6, 16, 5, 
           conv3_input, conv3_weight, conv3_bias, conv3_output> conv3("3");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(conv1.init(), "init conv1");
  adfCheck(conv1.run(1), "run conv1");
	adfCheck(conv1.end(), "end conv1");

  adfCheck(conv3.init(), "init conv3");
  adfCheck(conv3.run(1), "run conv3");
	adfCheck(conv3.end(), "end conv3");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(conv1.init(), "init conv1");
  get_graph_throughput_by_port(conv1, "plout[0]", conv1.plout[0], 1*24*24*6, sizeof(float32), ITER_CNT);
	adfCheck(conv1.end(), "end conv1");

  adfCheck(conv3.init(), "init conv3");
  get_graph_throughput_by_port(conv3, "plout[0]", conv3.plout[0], 1*8*8*16, sizeof(float32), ITER_CNT);
	adfCheck(conv3.end(), "end conv3");

  return 0;
}
#endif
