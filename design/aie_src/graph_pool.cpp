#include "graph_pool.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
char pool1_input[]  = "lenet_mnist__2__.pool1.MaxPool__.relu1.Relu_output_0__1x24x24x6.txt";
char pool1_output[] = "lenet_mnist__2__.pool1.MaxPool__.pool1.MaxPool_output_0__1x12x12x6.txt";
MaxpoolScalar<24, 12, 1, 6, pool1_input, pool1_output> pool1("1");

char pool2_input[]  = "lenet_mnist__5__.pool2.MaxPool__.relu2.Relu_output_0__1x8x8x16.txt";
char pool2_output[] = "lenet_mnist__5__.pool2.MaxPool__.pool2.MaxPool_output_0__1x4x4x16.txt";
MaxpoolScalar<8, 4, 1, 16, pool2_input, pool2_output> pool2("2");


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(pool1.init(), "init pool1");
  adfCheck(pool1.run(1), "run pool1");
	adfCheck(pool1.end(), "end pool1");

  adfCheck(pool2.init(), "init pool2");
  adfCheck(pool2.run(1), "run pool2");
	adfCheck(pool2.end(), "end pool2");
  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(pool1.init(), "init pool1");
  get_graph_throughput_by_port(pool1, "plout[0]", pool1.plout[0], 1*12*12*6, sizeof(float32), ITER_CNT);
	adfCheck(pool1.end(), "end pool1");

  adfCheck(pool2.init(), "init pool2");
  get_graph_throughput_by_port(pool2, "plout[0]", pool2.plout[0], 1*4*4*16, sizeof(float32), ITER_CNT);
	adfCheck(pool2.end(), "end pool2");

  return 0;
}
#endif
