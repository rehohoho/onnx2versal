import argparse
import os
import sys

import numpy as np

DATA_DIR = "data/"
OUT_DIR = "reports_dir/net_x1/sw_emu/x86simulator_output"


def load_txt(filepath: str):
  with open(filepath, "r") as f:
    raw = f.read().split("\n")
  data = []
  for line in raw:
    if "TLAST" not in line and "T " not in line and line != "":
      data.append([float(i) for i in line.split()])
  return np.array(data)


def check(filepath1: str,
          filepath2: str):
  arr1 = load_txt(filepath1)
  arr2 = load_txt(filepath2)
  assert arr1.shape ==  arr2.shape
  np.testing.assert_allclose(arr1, arr2, rtol=1e-03, atol=1e-05)
  print(f"TEST: ok! (shapes match and within rtol=1e-03, atol=1e-05)")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog=sys.argv[0], 
                                   description="Compare diff between two files.")
  parser.add_argument("-f1")
  parser.add_argument("-f2")
  args = parser.parse_args()

  if args.f1 and args.f2 and os.path.isfile(args.f1) and os.path.isfile(args.f2):
    print(f"Checking {args.f1} against {args.f2}")
    check(args.f1, args.f2)
  else:
    if args.f1 and args.f2 and os.path.isdir(args.f1) and os.path.isdir(args.f2):
      DATA_DIR = args.f1
      OUT_DIR = args.f2
    
    print(f"Checking directories {DATA_DIR} and {OUT_DIR}")
    filenames = set(os.listdir(DATA_DIR)).intersection(os.listdir(OUT_DIR))
    for i, fn in enumerate(filenames):
      print(f"Checking {i+1}/{len(filenames)}: {fn}")
      check(
        os.path.join(DATA_DIR, fn), 
        os.path.join(OUT_DIR, fn))
