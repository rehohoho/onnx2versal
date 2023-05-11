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
    ("plin0_lenet_input", f"{args.input_dir}/mnist_test_data.txt", 64, "float32")
  ]

  slave_list = [
    ("plout0_lenet_argm19", f"{args.output_dir}/mnist_test_label.txt", 64, "float32", 100)
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
