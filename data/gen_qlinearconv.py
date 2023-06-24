import math

import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, get_shape_str, save_tensor, round_away

np.random.seed(0)
VECTOR_WORD_BOUNDARY = 16 # in bytes


INP_H = 26 # for 3x3 stride 2 to be not padded
INP_W = 28
B = 1
M = 6
C = 1 # loop dependency missed issue occurs at C=1

X_scale = 0.004
W_scale = 0.003
Y_scale = 0.002
X_zero_point = 25
W_zero_point = 0
Y_zero_point = 19

tin = np.arange(C*INP_H*INP_W).astype(np.int8).reshape(1,C,INP_H,INP_W)
tbias = (np.arange(M) / (X_scale*W_scale/Y_scale)).astype(np.int32)
save_tensor("qlinearconv_int8in.txt", tin)
save_tensor("qlinearconv_int8in_pad.txt", pad_lastdim(tin, "qlinearconv tin", get_vector_boundary(tin), value=X_zero_point))
print("int8bias\n", tbias.flatten().tolist(), "\n\n\n")


# 5x5
K = 5

tw = np.arange(M*C*K*K).astype(np.int8).reshape(M,C,K,K)
print("int8weights_5x5\n", tw.flatten().tolist(), "\n\n\n")


# tin = pad_lastdim(tin, "tin", get_vector_boundary(tin))
tw = pad_lastdim(tw, "tw", get_vector_boundary(tw))
if K == 5:
  tw = tw[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]
print("int8weights_5x5_pad\n", tw.flatten().tolist(), "\n\n\n")


Y = torch.nn.functional.conv2d(
  torch.Tensor(tin.astype(int) - X_zero_point),
  torch.Tensor(tw[:,:,:,(4,6,8,10,12)].astype(int) - W_zero_point),
  torch.Tensor(tbias[:]), padding="same").numpy() * X_scale*W_scale/Y_scale
Y = round_away(Y) + Y_zero_point
Y = np.clip(Y, -128, 127).astype(np.int8)
save_tensor(f"qlinearconv_int8out_{get_shape_str(Y)}.txt", Y)


# Fixed point arithmetic implementation
scale = X_scale*W_scale/Y_scale
scalebits = abs(int(math.log(scale, 2))) + 16
Y_fix = Y.astype(int) * int(scale * 2**scalebits)
Y_fix = ((Y_fix + 2**(scalebits - 1)) >> scalebits) + Y_zero_point
Y_fix = np.clip(Y, -128, 127).astype(np.int8)
assert np.all(Y == Y_fix)


# 3x3
K = 3
PAD = (16 - K*K%16) % 16
tw_3x3 = np.arange(M*C*K*K).astype(np.int8).reshape(M,C,K,K)
print("int8weights_3x3\n", tw_3x3.flatten().tolist(), "\n\n\n")


tw_3x3_pad = np.pad(tw_3x3.reshape(M,C,-1), ((0,0),(0,0),(0,PAD)), 
                    "constant", constant_values=0)
print("int8weights_3x3_pad\n", tw_3x3_pad.flatten().tolist(), "\n\n\n")


Y_3x3 = torch.nn.functional.conv2d(
  torch.Tensor(tin.astype(int) - X_zero_point),
  torch.Tensor(tw_3x3.astype(int) - W_zero_point),
  torch.Tensor(tbias[:]), padding="same").numpy() * X_scale*W_scale/Y_scale
Y_3x3 = round_away(Y_3x3) + Y_zero_point
Y_3x3 = np.clip(Y_3x3, -128, 127).astype(np.int8)
save_tensor(f"qlinearconv_int8out_3x3_{get_shape_str(Y_3x3)}.txt", Y_3x3)


Y_3x3_stride2 = torch.nn.functional.conv2d(
  torch.Tensor(tin.astype(int) - X_zero_point),
  torch.Tensor(tw_3x3.astype(int) - W_zero_point),
  torch.Tensor(tbias[:]), stride=2).numpy() * X_scale*W_scale/Y_scale
Y_3x3_stride2 = round_away(Y_3x3_stride2) + Y_zero_point
Y_3x3_stride2 = np.clip(Y_3x3_stride2, -128, 127).astype(np.int8)
save_tensor(f"qlinearconv_int8out_3x3_stride2_{get_shape_str(Y_3x3_stride2)}.txt", Y_3x3_stride2)
