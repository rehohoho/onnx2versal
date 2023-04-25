import numpy as np
from math import *
import sys
import argparse

def get_time_ns(stamp):
  time_ns = float(stamp[0])
  if(stamp[1] == 'ps'):
    time_ns = time_ns/1000.0
  elif(stamp[1] == 'us'):
    time_ns = time_ns*1000.0
  elif(stamp[1] == 'ms'):
    time_ns = time_ns*1000000.0
  elif(stamp[1] == 's'):
    time_ns = time_ns*1000000000.0
  return(time_ns)


def read_file(filename):
  with open(filename, 'r') as f:
    raw = f.read().split("T ")
    
  data = []
  for line in raw:
    if line == "": continue
    
    # ['3296 ns', 'TLAST', '64 64 64 64 64 64 64 64 ', '']
    # ['3251200 ps', '64 64 64 64 64 64 64 64 ', '']
    line = line.split("\n")
    if not len(line) == 3 and not len(line) == 4:
      raise ValueError(f"Unexpected line in {filename}: {line}")

    data.append([
      get_time_ns(line[0].split()),
      *(float(i) for i in line[-2].split()),
      int(line[1] == "TLAST")
    ])
    
  return np.array(data)


def get_throughput(filename, is_complex):
  V = read_file(filename)
  full_frames = int(V[:,-1].sum())
  print(f"\n==============================\n{filename}\n")  
  print(f"Number of Full Frames: {full_frames}")

  # Basic Throughput computation
  if is_complex:
    ratio = 0.5
  else:
    ratio = 1
  
  nrows, ncols = V.shape
  raw_count = nrows * (ncols-2)
  raw_e2e = V[-1,0] - V[0,0]
  raw_throughput_msps = float(raw_count)/raw_e2e * ratio * 1000.0
  print(f"Raw Count: {raw_count}")
  print(f"Raw End-to-end time: {raw_e2e:.2f} ns")
  print(f"Raw Throughput: {raw_throughput_msps:.2f} msps")

  # If the output is frame based, compute a more precise throughput
  tlast = np.where(V[:,-1] == 1.0)
  if(len(tlast[0]) > 1):
    tlast = tlast[0]
    end_row = tlast[len(tlast)-2]+1
    # end_row is the number of Rows I take into account for the number of datasource
    # The timestamp I am interested in is the timestamp of the next transaction
    frame_count = end_row * (ncols - 2)
    frame_e2e = V[end_row, 0] - V[0, 0]
    frame_throughput_msps = float(frame_count)/frame_e2e * ratio * 1000.0
    print(f"Frame Count: {frame_count}")
    print(f"Frame End-to-end time: {frame_e2e:.2f} ns")
    print(f"Frame Throughput: {frame_throughput_msps:.2f} msps")

  print("\n")

# Entry point of this file
if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog=sys.argv[0], description='Compute the throughput corresponding to some output of AIE Simulations')
  parser.add_argument('--iscomplex', action='store_true', help='Indicates Complex data in the file')
  parser.add_argument('filename',nargs='+')
  args = sys.argv
  args.pop(0)
  args = parser.parse_args(args)
  print("This scripts assumes each output is one sample, e.g. 64 64 64 64 64 64 64 64 is counted as 8 samples.")
  print("Manually get throughput using #sample/(lastTimestamp - firstTimestamp) if that is not the case")

  for f in args.filename:
    get_throughput(f, args.iscomplex)
