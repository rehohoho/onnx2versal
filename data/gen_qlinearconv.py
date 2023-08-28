import math

import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, get_shape_str, save_tensor, round_away

np.random.seed(0)
VECTOR_WORD_BOUNDARY = 16 # in bytes


def print_cpp_vector(tensor: np.ndarray, name: str, dtype: str = "TTPARAM"):
  print(f"\nstd::vector<{dtype}> {name} {{{', '.join(str(i) for i in tensor.flatten().tolist())}}};\n")

INP_H = 26 # for 3x3 stride 2 to be not padded
INP_W = 28
B = 1
M = 4
C = 3 # note loop dependency missed issue occurs at C=1

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


# 5x5
K = 5

tw_5x5 = np.arange(M*C*K*K).astype(np.int8).reshape(M,C,K,K)
print_cpp_vector(tw_5x5, name="int8weights_5x5")

tb_5x5 = tbias - tw_5x5.reshape(M,-1).sum(1) * X_zero_point
print_cpp_vector(tb_5x5, name="int8bias_5x5", dtype="int32_t")

tw_5x5 = pad_lastdim(tw_5x5, "tw_5x5", get_vector_boundary(tw_5x5))
if K == 5:
  tw_5x5 = tw_5x5[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]
print_cpp_vector(tw_5x5, name="int8weights_5x5_pad")

Y = torch.nn.functional.conv2d(
  torch.Tensor(tin.astype(int) - X_zero_point),
  torch.Tensor(tw_5x5[:,:,:,(4,6,8,10,12)].astype(int) - W_zero_point),
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
print_cpp_vector(tw_3x3, name="int8weights_3x3")

tb_3x3 = tbias - tw_3x3.reshape(M,-1).sum(1) * X_zero_point
print_cpp_vector(tb_3x3, name="int8bias_3x3", dtype="int32_t")

tw_3x3_pad = pad_lastdim(tw_3x3.reshape(M,C,-1), "qlinearconv tw_3x3 int16int8", get_vector_boundary(tw_3x3))
print_cpp_vector(tw_3x3_pad, name="int8weights_3x3_pad")

tw_3x3_int16int8 = tw_3x3_pad[..., [0,1,2,9, 3,4,5,9, 6,7,8,9, 9,9,9,9]]
print_cpp_vector(tw_3x3_int16int8, name="int8weights_3x3_int16int8mac")

tw_3x3_int8int8 = pad_lastdim(tw_3x3, "qlinearconv tw_3x3 int8int8", get_vector_boundary(tw_3x3))
tw_3x3_int8int8 = tw_3x3_int8int8[..., [15,15,15,15, 0,0,1,1, 2,2,15,15, 15,15,15,15]]
print_cpp_vector(tw_3x3_int8int8, name="int8weights_3x3_int8int8mac")

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


# 1x1
tw_1x1 = np.arange(M*C).astype(np.int8).reshape(M,C,1,1)
print_cpp_vector(tw_1x1, name="int8weights_1x1")
tb_1x1 = tbias - tw_1x1.reshape(M,-1).sum(1) * X_zero_point
print_cpp_vector(tb_1x1, name="int8bias_1x1", dtype="int32_t")
tw_1x1_pad = pad_lastdim(tw_1x1.reshape(M,-1), "qlinearconv tw_1x1", get_vector_boundary(tw_1x1))
print_cpp_vector(tw_1x1_pad, name="int8weights_1x1_pad")

Y_1x1 = torch.nn.functional.conv2d(
  torch.Tensor(tin.astype(int) - X_zero_point),
  torch.Tensor(tw_1x1.astype(int) - W_zero_point),
  torch.Tensor(tbias[:]), padding="same").numpy() * X_scale*W_scale/Y_scale
Y_1x1 = round_away(Y_1x1) + Y_zero_point
Y_1x1 = np.clip(Y_1x1, -128, 127).astype(np.int8)
save_tensor(f"qlinearconv_int8out_1x1_{get_shape_str(Y_1x1)}.txt", Y_1x1)

Y_1x1_stride2 = torch.nn.functional.conv2d(
  torch.Tensor(tin.astype(int) - X_zero_point),
  torch.Tensor(tw_1x1.astype(int) - W_zero_point),
  torch.Tensor(tbias[:]), stride=2).numpy() * X_scale*W_scale/Y_scale
Y_1x1_stride2 = round_away(Y_1x1_stride2) + Y_zero_point
Y_1x1_stride2 = np.clip(Y_1x1_stride2, -128, 127).astype(np.int8)
save_tensor(f"qlinearconv_int8out_1x1_stride2_{get_shape_str(Y_1x1_stride2)}.txt", Y_1x1_stride2)
