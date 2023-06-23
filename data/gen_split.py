import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, get_shape_str, save_tensor
np.random.seed(0)

H = 10
INP_W = 64
OUT_W = 22 # (INP_W - OVERLAP) % (OUT_W - OVERLAP) == 0
OVERLAP = 1

fpin = np.random.random(size=(H, INP_W)).astype(np.float32)
save_tensor("split_fpin.txt", fpin)

stride = OUT_W - OVERLAP
for i in range(0, INP_W - stride, stride):
  print(i, i+OUT_W)
  tensor = fpin[:,i:i+OUT_W]
  save_tensor(f"split_fpout{i//stride}_{get_shape_str(tensor)}.txt", tensor)

OUT_W = 31 # (INP_W - OUT_W) % (OUT_W - OVERLAP) == 0
OVERLAP = -1

stride = OUT_W - OVERLAP
for i in range(0, INP_W, stride):
  print(i, i+OUT_W)
  tensor = fpin[:,i:i+OUT_W]
  save_tensor(f"split_fpout{i//stride}_{get_shape_str(tensor)}.txt", tensor)


H = 10
INP_W = 160
OUT_W = 64 # (INP_W - OVERLAP) % (OUT_W - OVERLAP) == 0
OVERLAP = 16

int8in = np.random.randint(-128, 128, size=(H, INP_W)).astype(np.int8)
save_tensor(f"split_int8in_{get_shape_str(int8in)}.txt", int8in)

stride = OUT_W - OVERLAP
for i in range(0, INP_W - stride, stride):
  print(i, i+OUT_W)
  tensor = int8in[:,i:i+OUT_W]
  save_tensor(f"split_int8out{i//stride}_{get_shape_str(tensor)}.txt", tensor)

OUT_W = 64 # (INP_W - OUT_W) % (OUT_W - OVERLAP) == 0
OVERLAP = -32

stride = OUT_W - OVERLAP
for i in range(0, INP_W, stride):
  print(i, i+OUT_W)
  tensor = int8in[:,i:i+OUT_W]
  save_tensor(f"split_int8out{i//stride}_neg_{get_shape_str(tensor)}.txt", tensor)