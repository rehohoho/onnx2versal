#include "graph_lenet.h"
#include "graph_utils.h"

#define ITER_CNT 1


// instance to be compiled and used in host within xclbin
char inp0[]    = "lenet_mnist__0___conv1_Conv__input__1x28x28x1.txt";

char conv1_w[]  = "lenet_mnist__0___conv1_Conv__conv1_weight__6x5x5x1.txt";
char conv1_b[]  = "lenet_mnist__0___conv1_Conv__conv1_bias__6.txt";
char conv3_w[]  = "lenet_mnist__3___conv2_Conv__conv2_weight__16x5x5x6.txt";
char conv3_b[]  = "lenet_mnist__3___conv2_Conv__conv2_bias__16.txt";

char gemm14_w[] = "lenet_mnist__14___fc1_Gemm__fc1_weight__120x256.txt";
char gemm14_b[] = "lenet_mnist__14___fc1_Gemm__fc1_bias__120.txt";
char gemm16_w[] = "lenet_mnist__16___fc2_Gemm__fc2_weight__84x120.txt";
char gemm16_b[] = "lenet_mnist__16___fc2_Gemm__fc2_bias__84.txt";
char gemm18_w[] = "lenet_mnist__18___fc3_Gemm__fc3_weight__10x84.txt";
char gemm18_b[] = "lenet_mnist__18___fc3_Gemm__fc3_bias__10.txt";

char o_conv00[] = "lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x24x24x6.txt";
char o_pool02[] = "lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x12x12x6.txt";
char o_conv03[] = "lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x8x8x16.txt";
char o_pool05[] = "lenet_mnist__5___pool2_MaxPool___pool2_MaxPool_output_0__1x4x4x16.txt";
char o_tran05[] = "lenet_mnist__13___Reshape___Reshape_output_0__1x256.txt";
char o_gemm14[] = "lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt";
char o_gemm16[] = "lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt";
char o_gemm18[] = "lenet_mnist__19___relu5_Relu__output__1x10.txt";

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
