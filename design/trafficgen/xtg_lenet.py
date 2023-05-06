import argparse
import logging

from xtg_aie import ExternalTraffic

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='xtg_lenet.py')
  parser.add_argument('--input_dir', required=True)
  parser.add_argument('--output_dir', required=True)
  args = parser.parse_args()
  
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

  master_list = [
    # ("plin0_lenet_input", f"{args.input_dir}/lenet_mnist__0___conv1_Conv__input__1x1x28x28.txt", 64, "float32")
    ("plin0_lenet_input", f"{args.input_dir}/mnist_test_data.txt", 64, "float32")
  ]

  slave_list = [
    # ("plout0_lenet_conv00", f"{args.output_dir}/lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt", 64, "float32", 24*24*6),
    # ("plout1_lenet_pool01", f"{args.output_dir}/lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt", 64, "float32", 12*12*6),
    # ("plout2_lenet_conv02", f"{args.output_dir}/lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt", 64, "float32", 8*8*16),
    # ("plout3_lenet_pool03", f"{args.output_dir}/lenet_mnist__5___pool2_MaxPool___pool2_MaxPool_output_0__1x16x4x4.txt", 64, "float32", 4*4*16),
    # ("plout4_lenet_gemm14", f"{args.output_dir}/lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt", 64, "float32", 120),
    # ("plout5_lenet_gemm16", f"{args.output_dir}/lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt", 64, "float32", 84),
    # ("plout6_lenet_gemm18", f"{args.output_dir}/lenet_mnist__19___relu5_Relu__output__1x10.txt", 64, "float32", 10),
    # ("plout7_lenet_argm19", f"{args.output_dir}/lenet_out.txt", 64, "float32", 1)
    ("plout0_lenet_argm19", f"{args.output_dir}/mnist_test_label.txt", 64, "float32", 100)
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
