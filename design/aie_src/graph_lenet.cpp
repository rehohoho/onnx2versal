#include "graph_lenet.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
char inp0[]    = "lenet_mnist__0__.conv1.Conv__input__1x28x28x1.txt";

char conv1_w[]  = "lenet_mnist__0__.conv1.Conv__conv1.weight__6x5x5x1.txt";
char conv1_b[]  = "lenet_mnist__0__.conv1.Conv__conv1.bias__6.txt";
char conv3_w[]  = "lenet_mnist__3__.conv2.Conv__conv2.weight__16x5x5x6.txt";
char conv3_b[]  = "lenet_mnist__3__.conv2.Conv__conv2.bias__16.txt";

char gemm14_w[] = "lenet_mnist__14__.fc1.Gemm__fc1.weight__120x256.txt";
char gemm14_b[] = "lenet_mnist__14__.fc1.Gemm__fc1.bias__120.txt";
char gemm16_w[] = "lenet_mnist__16__.fc2.Gemm__fc2.weight__84x120.txt";
char gemm16_b[] = "lenet_mnist__16__.fc2.Gemm__fc2.bias__84.txt";
char gemm18_w[] = "lenet_mnist__18__.fc3.Gemm__fc3.weight__10x84.txt";
char gemm18_b[] = "lenet_mnist__18__.fc3.Gemm__fc3.bias__10.txt";

char o_conv00[] = "lenet_mnist__1__.relu1.Relu__.relu1.Relu_output_0__1x24x24x6.txt";
char o_pool02[] = "lenet_mnist__2__.pool1.MaxPool__.pool1.MaxPool_output_0__1x12x12x6.txt";
char o_conv03[] = "lenet_mnist__4__.relu2.Relu__.relu2.Relu_output_0__1x8x8x16.txt";
char o_pool05[] = "lenet_mnist__5__.pool2.MaxPool__.pool2.MaxPool_output_0__1x4x4x16.txt";
char o_tran05[] = "lenet_mnist__13__.Reshape__.Reshape_output_0__1x256.txt";
char o_gemm14[] = "lenet_mnist__15__.relu3.Relu__.relu3.Relu_output_0__1x120.txt";
char o_gemm16[] = "lenet_mnist__17__.relu4.Relu__.relu4.Relu_output_0__1x84.txt";
char o_gemm18[] = "lenet_mnist__19__.relu5.Relu__output__1x10.txt";

MnistLenetScalar< 
  inp0,
  conv1_w, conv1_b,
  conv3_w, conv3_b,
  gemm14_w, gemm14_b,
  gemm16_w, gemm16_b,
  gemm18_w, gemm18_b
> lenet1 ("1", 
  o_conv00, o_pool02, o_conv03, o_pool05, o_tran05, o_gemm14, o_gemm16, o_gemm18);


#ifdef __X86SIM__
int main(int argc, char ** argv) {
	adfCheck(lenet1.init(), "init lenet1");
  adfCheck(lenet1.run(1), "run lenet1");
	adfCheck(lenet1.end(), "end lenet1");

  return 0;
}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {
	
	adfCheck(lenet1.init(), "init lenet1");
  get_graph_throughput_by_port(lenet1, "plout[0]", lenet1.plout[0], 1*10, sizeof(float32), ITER_CNT);
	adfCheck(lenet1.end(), "end lenet1");

  return 0;
}
#endif
