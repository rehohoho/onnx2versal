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
    ("plin0_concat1_input", f"{args.input_dir}/concat_in.txt", 64, "float32"),
    ("plin1_concat1_input", f"{args.input_dir}/concat_in.txt", 64, "float32"),
    ("plin2_concat1_input", f"{args.input_dir}/concat_in.txt", 64, "float32"),
    ("plin3_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
  ]

  slave_list = [
    ("plout0_concat1_output", f"{args.output_dir}/concat_out.txt", 64, "float32", 18),
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
