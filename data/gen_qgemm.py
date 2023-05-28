import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, \
  save_tensor, round_away

np.random.seed(0)


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
vector_size = get_vector_boundary(tw)
tw = pad_lastdim(tw, "QGemm weights", vector_size)
tbias = pad_lastdim(tbias, "QGemm bias", vector_size)

tout = torch.nn.functional.linear(
  torch.Tensor(tin.astype(int) - tin_zero),
  torch.Tensor(tw.T.astype(int) - tw_zero), #NxK
  torch.Tensor(tbias[:])).numpy() * tin_scale*tw_scale/tout_scale
tout = round_away(tout) + tout_zero
tout = np.clip(tout, -128, 127).astype(np.int8)

# save
save_tensor("qgemm_int8in.txt", tin)
save_tensor("qgemm_int8out_qgemmScalar_shape1x10.txt", tout)
save_tensor("qgemm_int8out_qgemmVector_shape1x10.txt", tout)
print("int8weights\n", tw.flatten().tolist(), "\n\n\n")
print("int8bias\n", tbias.flatten().tolist(), "\n\n\n")
