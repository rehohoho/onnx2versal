import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor, round_away

np.random.seed(0)


INP_H = 28
INP_W = 28
OUT_W = 32 # padded
SCALE = 0.00392156889
ZERO = -128

tin = np.random.random(size=(INP_H,INP_W)).astype(np.float32)
tout = round_away(tin / SCALE) + ZERO
tout = np.clip(tout, -128, 127).astype(np.int8)

save_tensor("quantizelinear_int8in.txt", tin)
save_tensor(f"quantizelinear_int8out_shape{INP_H}x{INP_W}.txt", tout)
print("SCALE\n", SCALE, "\n\n\n")
print("ZERO\n", ZERO, "\n\n\n")
