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
fp2streamout = [[], []]
for i in range(0, INP_W - stride, stride):
  print(i, i+OUT_W)
  tensor = fpin[:,i:i+OUT_W]
  save_tensor(f"split_fpout{i//stride}_{get_shape_str(tensor)}.txt", tensor)
  if len(fp2streamout[0]) <= len(fp2streamout[1]):
    fp2streamout[0].append(tensor)
  else:
    fp2streamout[1].append(tensor)

fpout1 = np.hstack(fp2streamout[0])
save_tensor(f"split_fpout0_2stream_{get_shape_str(fpout1)}.txt", fpout1)
fpout2 = np.hstack(fp2streamout[1])
save_tensor(f"split_fpout1_2stream_{get_shape_str(fpout2)}.txt", fpout2)

OUT_W = 31 # (INP_W - OUT_W) % (OUT_W - OVERLAP) == 0
OVERLAP = -1

stride = OUT_W - OVERLAP
fp2streamout = [[], []]
for i in range(0, INP_W, stride):
  print(i, i+OUT_W)
  tensor = fpin[:,i:i+OUT_W]
  save_tensor(f"split_fpout{i//stride}_{get_shape_str(tensor)}.txt", tensor)
  if len(fp2streamout[0]) <= len(fp2streamout[1]):
    fp2streamout[0].append(tensor)
  else:
    fp2streamout[1].append(tensor)

fpout1_neg = np.hstack(fp2streamout[0])
save_tensor(f"split_fpout0_neg2stream_{get_shape_str(fpout1_neg)}.txt", fpout1_neg)
fpout2_neg = np.hstack(fp2streamout[1])
save_tensor(f"split_fpout1_neg2stream_{get_shape_str(fpout2_neg)}.txt", fpout2_neg)

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