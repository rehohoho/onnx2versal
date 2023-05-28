import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


CHUNK_CNT = 24
CHUNK_SIZE = 23
SCALE = 0.00392156889
ZERO = -128

tin = np.random.randint(-128, 128, size=CHUNK_CNT*CHUNK_SIZE).reshape(CHUNK_CNT, CHUNK_SIZE).astype(np.int8)
tout = ((tin.astype(int) - ZERO) * SCALE).astype(np.float32)

tin = pad_lastdim(tin, "tin dequantize_linear", get_vector_boundary(tin))
save_tensor("dequantizelinear_int8in.txt", tin)
save_tensor("dequantizelinear_fpout_DequantizeLinearScalar_shape24x23.txt", tout)
print("SCALE\n", SCALE, "\n\n\n")
print("ZERO\n", ZERO, "\n\n\n")
