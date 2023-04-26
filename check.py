import argparse
import sys

import numpy as np


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog=sys.argv[0], description='Compare diff between two files.')
  parser.add_argument('f1')
  parser.add_argument('f2')
  args = sys.argv
  args.pop(0)
  args = parser.parse_args(args)

  arr1 = np.loadtxt(args.f1)
  arr2 = np.loadtxt(args.f2)
  assert arr1.shape ==  arr2.shape
  np.testing.assert_allclose(arr1, arr2, rtol=1e-03, atol=1e-05)
  print(f"TEST: ok! (shapes match and within rtol=1e-03, atol=1e-05)")