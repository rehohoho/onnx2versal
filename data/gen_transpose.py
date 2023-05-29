import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)

B = 1
H = 4
W = 4
C = 16

fpin = np.random.random(size=(B,H,W,C)).astype(np.float32)
fpout = fpin.transpose(0,3,1,2)

save_tensor("transpose_fpin.txt", fpin)
save_tensor(f"transpose_fpout_shape{B}x{C}x{H}x{W}.txt", fpout)
