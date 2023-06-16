import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, get_shape_str, save_tensor


H = 10
INP_W = 64
OUT_W = 22 # % INP_W - OVERLAP
OVERLAP = 1

fpin = np.random.random(size=(H, INP_W)).astype(np.float32)
save_tensor("split_fpin.txt", fpin)

stride = OUT_W - OVERLAP
for i in range(0, INP_W - stride, stride):
  print(i, i+OUT_W)
  tensor = fpin[:,i:i+OUT_W]
  save_tensor(f"split_fpout{i//stride}_{get_shape_str(tensor)}.txt", tensor)

OUT_W = 31 # % INP_W - OVERLAP
OVERLAP = -1

stride = OUT_W - OVERLAP
for i in range(0, INP_W - stride + 1, stride):
  print(i, i+OUT_W)
  tensor = fpin[:,i:i+OUT_W]
  save_tensor(f"split_fpout{i//stride}_{get_shape_str(tensor)}.txt", tensor)
