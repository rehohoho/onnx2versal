import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


INP_H = 24
INP_W = 24
INP_W_PAD = 32
OUT_W = 12
OUT_W_PAD = 16
B = 1
C = 6

# float32
fpin = np.random.random(size=(B,C,INP_H,INP_W)).astype(np.float32)
save_tensor("pool_fpin.txt", fpin)

tout = torch.nn.functional.max_pool2d(torch.Tensor(fpin), INP_W//OUT_W).numpy()
save_tensor("pool_fpout_MaxpoolScalarBCHW_shape1x6x12x12.txt", tout)
save_tensor("pool_fpout_Maxpool2x2FloatBCHW_shape1x6x12x12.txt", tout)

# int8
int8in = np.random.randint(
  -128, 128, size=(C*INP_H*INP_W)).reshape(B,C,INP_H,INP_W).astype(np.int8)
tout = torch.nn.functional.max_pool2d(
  torch.Tensor(int8in), INP_W//OUT_W).numpy().astype(np.int8)
int8in_pad = pad_lastdim(int8in, "int8in pad", get_vector_boundary(int8in))
save_tensor("pool_int8in_pad.txt", int8in_pad)
save_tensor(f"pool_int8out_Maxpool2x2Int8BCHW_shape{B}x{C}x{OUT_W}x{OUT_W}.txt", tout)

# float32 bhwc
fpin_bhwc = fpin.reshape(B,INP_H,INP_W,C)
tout_bhwc = torch.nn.functional.max_pool2d(
  torch.Tensor(fpin_bhwc.transpose(0,3,1,2)), INP_W//OUT_W).numpy().transpose(0,2,3,1)
save_tensor("pool_fpout_MaxpoolScalarBHWC_shape1x6x12x12.txt", tout_bhwc)
