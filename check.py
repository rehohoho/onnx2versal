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


def is_file_match(filepath1: str,
                  filepath2: str,
                  is_throw_err: int):
  arr1 = load_txt(filepath1)
  arr2 = load_txt(filepath2)

  if "shape" in filepath1:
    shape = os.path.splitext(filepath1)[0].split("shape")[-1].split("x")
    shape = [int(i) for i in shape]
    arr1 = arr1.reshape(*shape[:-1], -1)[..., :shape[-1]]
    arr2 = arr2.reshape(*shape[:-1], -1)[..., :shape[-1]]
  
  if arr1.shape == arr2.shape:
    if np.allclose(arr1, arr2, rtol=1e-03, atol=1e-05):
      print(f"TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)")
      return True
    else:
      print(f"TEST (tolerance): FAILED! (rtol=1e-03, atol=1e-05)")
      if is_throw_err == 0: import ipdb;ipdb.set_trace()
  
  else:
    print(f"WARNING: arr1 shape {arr1.shape}, arr2 shape {arr2.shape}")
    minSize = min(arr1.size, arr2.size)
    if np.allclose(arr1.flatten()[:minSize], arr2.flatten()[:minSize], rtol=1e-03, atol=1e-05):
      print(f"TEST (tolerance): first {minSize} OK! (rtol=1e-03, atol=1e-05)")
    else:
      print(f"TEST (tolerance): first {minSize} FAILED! (rtol=1e-03, atol=1e-05)")
      if is_throw_err == 0: import ipdb;ipdb.set_trace()

  return False


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog=sys.argv[0], 
                                   description="Compare diff between two files.")
  parser.add_argument("-f1")
  parser.add_argument("-f2")
  parser.add_argument("-err", type=int)
  args = parser.parse_args()

  pass_count = 0

  if args.f1 and args.f2 and os.path.isfile(args.f1) and os.path.isfile(args.f2):
    print(f"Checking {args.f1} against {args.f2}")
    pass_count += is_file_match(args.f1, args.f2, args.err)
  else:
    if args.f1 and args.f2 and os.path.isdir(args.f1) and os.path.isdir(args.f2):
      DATA_DIR = args.f1
      OUT_DIR = args.f2
    
    print(f"Checking directories {DATA_DIR} and {OUT_DIR}")
    filenames = set(os.listdir(DATA_DIR)).intersection(os.listdir(OUT_DIR))

    for i, fn in enumerate(filenames):
      print(f"Checking {i+1}/{len(filenames)}: {fn}")
      pass_count += is_file_match(os.path.join(DATA_DIR, fn), 
                                  os.path.join(OUT_DIR, fn), 
                                  args.err)
  
  assert(pass_count == len(filenames)), f"{pass_count} / {len(filenames)} tests passed."
