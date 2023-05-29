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


def is_file_match(res_path: str,
                  data_path: str,
                  is_throw_err: int):
  res_arr = load_txt(res_path)
  data_arr = load_txt(data_path)

  if "shape" in data_path:
    shape = os.path.splitext(data_path)[0].split("shape")[-1].split("x")
    shape = [int(i) for i in shape]
    res_arr = res_arr.reshape(*shape[:-1], -1)[..., :shape[-1]]
    data_arr = data_arr.reshape(*shape[:-1], -1)[..., :shape[-1]]
  
  if res_arr.shape == data_arr.shape:
    if np.allclose(res_arr, data_arr, rtol=1e-03, atol=1e-05):
      print(f"TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)")
      return True
    else:
      print(f"TEST (tolerance): FAILED! (rtol=1e-03, atol=1e-05)")
      if is_throw_err == 0: import ipdb;ipdb.set_trace()
  
  else:
    print(f"WARNING: res_arr shape {res_arr.shape}, data_arr shape {data_arr.shape}")
    minSize = min(res_arr.size, data_arr.size)
    if np.allclose(res_arr.flatten()[:minSize], data_arr.flatten()[:minSize], rtol=1e-03, atol=1e-05):
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
      OUT_DIR = args.f1
      DATA_DIR = args.f2
    
    print(f"Checking directories out: {OUT_DIR} and data: {DATA_DIR}")
    res_filenames = os.listdir(OUT_DIR)
    
    filepairs = []
    for res_fn in res_filenames:
      data_basename = "_".join(res_fn.split("_")[:-1])+".txt"
      data_path = os.path.join(DATA_DIR, data_basename)
      if os.path.exists(data_path):
        filepairs.append((os.path.join(OUT_DIR, res_fn), data_path))

    for i, (res_fn, data_fn) in enumerate(filepairs):
      print(f"Checking {i+1}/{len(filepairs)}: {res_fn.replace(OUT_DIR, '')} against {data_fn.replace(DATA_DIR, '')}")
      pass_count += is_file_match(res_fn, data_fn, args.err)
  
  assert(pass_count == len(filepairs)), f"{pass_count} / {len(filepairs)} tests passed."
