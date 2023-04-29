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
    ("plin3_concat1_input", f"{args.input_dir}/concat_in.txt", 64, "float32"),
    ("plin4_concat1_input", f"{args.input_dir}/concat_in.txt", 64, "float32"),
    ("plin5_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin6_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin7_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin8_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin9_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin10_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin11_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin12_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin13_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin14_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin15_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin16_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin17_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin18_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin19_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin20_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin21_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin22_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin23_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin24_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin25_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin26_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin27_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin28_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin29_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin30_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
    ("plin31_concat1_input", f"{args.input_dir}/empty.txt", 64, "float32"),
  ]

  slave_list = [
    ("plout0_concat1_output", f"{args.output_dir}/concat_out.txt", 64, "float32", 36),
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
