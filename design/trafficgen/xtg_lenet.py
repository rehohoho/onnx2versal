import argparse
import logging

from xtg_aie import ExternalTraffic

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='xtg_aie.py')
  parser.add_argument('--input_dir', required=True)
  parser.add_argument('--output_dir', required=True)
  args = parser.parse_args()
  
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

  master_list = [
    ("plin00_1_inp", f"{args.input_dir}/lenet_mnist__0___conv1_Conv__input__1x28x28x1.txt", 64, "float32"),
    ("plin01_1_conv00w", f"{args.input_dir}/lenet_mnist__0___conv1_Conv__conv1_weight__6x5x5x1.txt", 64, "float32"),
    ("plin02_1_conv00b", f"{args.input_dir}/lenet_mnist__0___conv1_Conv__conv1_bias__6.txt", 64, "float32"),
    ("plin03_1_conv03w", f"{args.input_dir}/lenet_mnist__3___conv2_Conv__conv2_weight__16x5x5x6.txt", 64, "float32"),
    ("plin04_1_conv03b", f"{args.input_dir}/lenet_mnist__3___conv2_Conv__conv2_bias__16.txt", 64, "float32"),
    ("plin05_1_gemm14w", f"{args.input_dir}/lenet_mnist__14___fc1_Gemm__fc1_weight__120x256.txt", 64, "float32"),
    ("plin06_1_gemm14b", f"{args.input_dir}/lenet_mnist__14___fc1_Gemm__fc1_bias__120.txt", 64, "float32"),
    ("plin07_1_gemm16w", f"{args.input_dir}/lenet_mnist__16___fc2_Gemm__fc2_weight__84x120.txt", 64, "float32"),
    ("plin08_1_gemm16b", f"{args.input_dir}/lenet_mnist__16___fc2_Gemm__fc2_bias__84.txt", 64, "float32"),
    ("plin09_1_gemm18w", f"{args.input_dir}/lenet_mnist__18___fc3_Gemm__fc3_weight__10x84.txt", 64, "float32"),
    ("plin10_1_gemm18b", f"{args.input_dir}/lenet_mnist__18___fc3_Gemm__fc3_bias__10.txt", 64, "float32"),
  ]

  slave_list = [
    ("plout0_1_conv00", f"{args.output_dir}/lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x24x24x6.txt", 64, "float32", 24*24*6),
    ("plout1_1_pool02", f"{args.output_dir}/lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x12x12x6.txt", 64, "float32", 12*12*6),
    ("plout2_1_conv03", f"{args.output_dir}/lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x8x8x16.txt", 64, "float32", 8*8*16),
    ("plout3_1_pool05", f"{args.output_dir}/lenet_mnist__5___pool2_MaxPool___pool2_MaxPool_output_0__1x4x4x16.txt", 64, "float32", 4*4*16),
    ("plout4_1_tran05", f"{args.output_dir}/lenet_mnist__13___Reshape___Reshape_output_0__1x256.txt", 64, "float32", 256),
    ("plout5_1_gemm14", f"{args.output_dir}/lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt", 64, "float32", 120),
    ("plout6_1_gemm16", f"{args.output_dir}/lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt", 64, "float32", 84),
    ("plout7_1_gemm18", f"{args.output_dir}/lenet_mnist__19___relu5_Relu__output__1x10.txt", 64, "float32", 10),
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
