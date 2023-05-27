import numpy as np
import torch

np.random.seed(0)
VECTOR_WORD_BOUNDARY = 16 # in bytes


def pad_lastdim(tensor: np.ndarray, 
                tensor_name: str, 
                N: int,
                value: int = 0):
  lastdim = tensor.shape[-1]
  pad_size = (N - lastdim%N) % N
  if pad_size != 0:
    print(f"Padding {tensor_name} {tensor.shape} to {*tensor.shape[:-1], lastdim+pad_size}")
    pad_arr = (*((0,0) for _ in range(tensor.ndim-1)),(0,pad_size))
    tensor = np.pad(tensor, pad_arr, "constant", constant_values=value)
  return tensor


def get_vector_boundary(tensor: np.ndarray):
  return VECTOR_WORD_BOUNDARY // tensor.dtype.itemsize


INP_H = 24
INP_W = 24
INP_W_PAD = 32
OUT_W = 12
OUT_W_PAD = 16
B = 1
C = 6

# float32
tin = np.arange(C*INP_H*INP_W).reshape(B,C,INP_H,INP_W)
tin_rand = np.random.random(C*INP_H*INP_W).reshape(B,C,INP_H,INP_W)
np.savetxt("pool_fpin.txt", tin.reshape(-1, 2))
np.savetxt("pool_fpin_rand.txt", tin_rand.reshape(-1, 2))

tout = torch.nn.functional.max_pool2d(torch.Tensor(tin), INP_W//OUT_W)
np.savetxt("pool_fpout_MaxpoolScalarBCHW.txt", tout.reshape(-1, 2), fmt="%.9e")
np.savetxt("pool_fpout_Maxpool2x2FloatBCHW.txt", tout.reshape(-1, 2), fmt="%.9e")

tout_rand = torch.nn.functional.max_pool2d(torch.Tensor(tin_rand), INP_W//OUT_W)
np.savetxt("pool_fpout_MaxpoolScalarBCHW_rand.txt", tout_rand.reshape(-1, 2), fmt="%.9e")
np.savetxt("pool_fpout_Maxpool2x2FloatBCHW_rand.txt", tout_rand.reshape(-1, 2), fmt="%.9e")

# float32 bhwc
tin_bhwc = tin.reshape(B,INP_H,INP_W,C)
tout_bhwc = torch.nn.functional.max_pool2d(
  torch.Tensor(tin_bhwc.transpose(0,3,1,2)), INP_W//OUT_W).numpy().transpose(0,2,3,1)
np.savetxt("pool_fpout_MaxpoolScalarBHWC.txt", tout_bhwc.reshape(-1, 2), fmt="%.9e")

tin_bhwc_rand = tin_rand.reshape(B,INP_H,INP_W,C)
tout_bhwc_rand = torch.nn.functional.max_pool2d(
  torch.Tensor(tin_bhwc_rand.transpose(0,3,1,2)), INP_W//OUT_W).numpy().transpose(0,2,3,1)
np.savetxt("pool_fpout_MaxpoolScalarBHWC_rand.txt", tout_bhwc_rand.reshape(-1, 2), fmt="%.9e")


# int8
tin = np.random.randint(
  -128, 128, size=(C*INP_H*INP_W)).reshape(B,C,INP_H,INP_W).astype(np.int8)
tout = torch.nn.functional.max_pool2d(
  torch.Tensor(tin), INP_W//OUT_W).numpy().astype(np.int8)

tin = pad_lastdim(tin, "tin", get_vector_boundary(tin))
np.savetxt("pool_int8in.txt", tin.reshape(-1, 8), fmt="%d")
np.savetxt(f"pool_int8out_Maxpool2x2Int8BCHW_shape{B}x{C}x{OUT_W}x{OUT_W}.txt", tout.reshape(-1,8), fmt="%d")
