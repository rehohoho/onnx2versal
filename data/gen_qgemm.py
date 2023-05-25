import numpy as np
import torch

np.random.seed(0)
VECTOR_WORD_BOUNDARY = 16 # in bytes


def round_away(x):
  x = np.round(x, 3) # rounds 4.499996 and 4.496 to 4.5 first
  a = np.abs(x)
  b = np.floor(a) + np.floor(2*(a%1))
  return np.sign(x)*b


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


M = 1
K = 84
N = 10
tin_scale = 0.004
tw_scale = 0.003
tout_scale = 0.002
tin_zero = 25
tw_zero = 0
tout_zero = 19

tin = np.tile(np.arange(11), M*K//11+1).astype(np.int8)[:M*K].reshape(M,K)
tw = np.tile(np.arange(11), K*N//11+1).astype(np.int8)[:K*N].reshape(K,N)
tbias = (np.arange(N) / (tin_scale*tw_scale/tout_scale)).astype(np.int32)

# padding for vector read/write
vector_size = VECTOR_WORD_BOUNDARY // tin.dtype.itemsize
# tin = pad_lastdim(tin, "QGemm tin", vector_size, value=tin_zero)
# tw = pad_lastdim(tw, "QGemm weights", vector_size)
tw = pad_lastdim(tw, "QGemm weights", vector_size)
tbias = pad_lastdim(tbias, "QGemm bias", vector_size)

tout = torch.nn.functional.linear(
  torch.Tensor(tin.astype(int) - tin_zero),
  torch.Tensor(tw.T.astype(int) - tw_zero), #NxK
  torch.Tensor(tbias[:])).numpy() * tin_scale*tw_scale/tout_scale
tout = round_away(tout) + tout_zero
tout = np.clip(tout, -128, 127).astype(np.int8)

tout = pad_lastdim(tout, "QGemm tout", vector_size, value=tin_zero)

# save
tin = pad_lastdim(tin, "pad to write tin", 8, value=0)
np.savetxt("qgemm_int8in.txt", tin.reshape(-1, 8), fmt="%d")
np.savetxt("qgemm_int8out_qgemmScalar.txt", tout.reshape(-1, 8), fmt="%d")
np.savetxt("qgemm_int8out_qgemmVector.txt", tout.reshape(-1, 8), fmt="%d")
print("int8weights\n", tw.flatten().tolist(), "\n\n\n")
print("int8bias\n", tbias.flatten().tolist(), "\n\n\n")
