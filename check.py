import argparse
import math
import os
import re
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
      data += [float(i) for i in line.split()]
  return np.array(data)


def is_file_match(res_path: str,
                  data_path: str,
                  is_throw_err: int):
  res_arr = load_txt(res_path)
  data_arr = load_txt(data_path)

  if "shape" in data_path:
    shape = re.findall(r"(?<=shape).*?(?=[_,.])", data_path)[-1].split("x")
    shape = [int(i) for i in shape]
    res_arr = res_arr.reshape(*shape[:-1], -1)[..., :shape[-1]]
    if data_arr.size - math.prod(shape) < shape[-1]:
      data_arr = data_arr.flatten()[:math.prod(shape)]
    data_arr = data_arr.reshape(*shape[:-1], -1)[..., :shape[-1]]
  
  if res_arr.shape == data_arr.shape:
    close_count = np.isclose(res_arr, data_arr, rtol=1e-03, atol=1e-05).sum()
    if close_count == res_arr.size:
      print(f"TEST: OK!")
      return True
    else:
      error = np.abs(res_arr - data_arr)
      ref = np.abs(data_arr)
      nonzero = np.nonzero(ref)
      
      relerr = np.max(error[nonzero] / ref[nonzero])
      abserr = np.max(error[nonzero])
      print(f"TEST: FAILED! Only {close_count}/{res_arr.size} passed.")
      print(f"Max absolute difference: {abserr}\nMax relative difference: {relerr}")
      if is_throw_err == 0: import ipdb;ipdb.set_trace()
  
  else:
    print(f"TEST: FAILED! res_arr shape {res_arr.shape}, data_arr shape {data_arr.shape}")
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
    
    print(f"Checking directories out: {OUT_DIR} and data: {DATA_DIR} at tolerance @ rtol=1e-03, atol=1e-05")
    res_filenames = os.listdir(OUT_DIR)
    
    filepairs = []
    for res_fn in res_filenames:
      if os.path.exists(os.path.join(DATA_DIR, res_fn)):
        filepairs.append((res_fn, res_fn))
        continue
      data_basename = "_".join(res_fn.split("_")[:-1])+".txt"
      if os.path.exists(os.path.join(DATA_DIR, data_basename)):
        filepairs.append((res_fn, data_basename))
    
    filepairs.sort()
      
    for i, (res_fn, data_fn) in enumerate(filepairs):
      print(f"Checking {i+1}/{len(filepairs)}: {res_fn:<80} against {data_fn:<48}", end="\t")
      pass_count += is_file_match(os.path.join(OUT_DIR, res_fn), 
                                  os.path.join(DATA_DIR, data_fn), 
                                  args.err)
  
  assert(pass_count == len(filepairs)), f"{pass_count} / {len(filepairs)} tests passed."
