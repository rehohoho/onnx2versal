import numpy as np
import torch

np.random.seed(0)


INP_H = 28
INP_W = 28
OUT_W = 32
SCALE = 0.00392156889
ZERO = -128

def round_away(x):
  x = np.round(x, 3) # rounds 4.499996 and 4.496 to 4.5 first
  a = np.abs(x)
  b = np.floor(a) + np.floor(2*(a%1))
  return np.sign(x)*b

tin = np.random.random(INP_H*INP_W).reshape(INP_H, INP_W)
tout = round_away(tin / SCALE) + ZERO
tout = np.clip(tout, -128, 127).astype(np.int8)
tout = np.pad(tout, ((0,0), (0,OUT_W-INP_W)), "constant", constant_values=ZERO)

np.savetxt("quantizelinear_int8in.txt", tin.reshape(-1, 2))
np.savetxt("quantizelinear_int8out_QuantizeLinearScalar.txt", tout.reshape(-1, 8), fmt="%d")
np.savetxt("quantizelinear_int8out_QuantizeLinearVector.txt", tout.reshape(-1, 8), fmt="%d")
print("SCALE\n", SCALE, "\n\n\n")
print("ZERO\n", ZERO, "\n\n\n")
