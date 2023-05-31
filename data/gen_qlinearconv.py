import math

import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor, round_away

np.random.seed(0)
VECTOR_WORD_BOUNDARY = 16 # in bytes


INP_H = 28
INP_W = 28
OUT_H = 24
OUT_W = 24
B = 1
M = 6
C = 1 # loop dependency missed issue occurs at C=1
K = 5

X_scale = 0.004
W_scale = 0.003
Y_scale = 0.002
X_zero_point = 25
W_zero_point = 0
Y_zero_point = 19

tin = np.arange(C*INP_H*INP_W).astype(np.int8).reshape(1,C,INP_H,INP_W)
tw = np.arange(M*C*K*K).astype(np.int8).reshape(M,C,K,K)
tbias = (np.arange(M) / (X_scale*W_scale/Y_scale)).astype(np.int32)

tin = pad_lastdim(tin, "tin", get_vector_boundary(tin))
tw = pad_lastdim(tw, "tw", get_vector_boundary(tw))
if K == 5:
  tw = tw[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]

Y = torch.nn.functional.conv2d(
  torch.Tensor(tin[:,:,:,:28].astype(int) - X_zero_point),
  torch.Tensor(tw[:,:,:,(4,6,8,10,12)].astype(int) - W_zero_point),
  torch.Tensor(tbias[:])).numpy() * X_scale*W_scale/Y_scale
Y = round_away(Y) + Y_zero_point
Y = np.clip(Y, -128, 127).astype(np.int8)

# Fixed point arithmetic implementation
scale = X_scale*W_scale/Y_scale
scalebits = abs(int(math.log(scale, 2))) + 16
Y_fix = Y.astype(int) * int(scale * 2**scalebits)
Y_fix = ((Y_fix + 2**(scalebits - 1)) >> scalebits) + Y_zero_point
Y_fix = np.clip(Y, -128, 127).astype(np.int8)
assert np.all(Y == Y_fix)

save_tensor("qlinearconv_int8in.txt", tin)
save_tensor(f"qlinearconv_int8out_shape{B}x{M}x{OUT_H}x{OUT_W}.txt", Y)
print("int8weights\n", tw.flatten().tolist(), "\n\n\n")
print("int8bias\n", tbias.flatten().tolist(), "\n\n\n")
