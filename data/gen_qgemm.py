import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, \
  save_tensor, round_away

np.random.seed(0)


M = 1
K = 80
N = 30
tin_scale = 0.004
tw_scale = 0.003
tout_scale = 0.002
tin_zero = 25
tw_zero = 0
tout_zero = 19

tin = np.tile(np.arange(11), M*K//11+1).astype(np.int8)[:M*K].reshape(M,K)
tw = np.tile(np.arange(11), K*N//11+1).astype(np.int8)[:K*N].reshape(K,N)
tbias = (np.arange(N) / (tin_scale*tw_scale/tout_scale)).astype(np.int32)
tbias_shift = (tbias - tw.sum(0) * tin_zero).astype(np.int32)

# padding for vector read/write
tw = pad_lastdim(tw, "QGemm weights", get_vector_boundary(tw))
tbias = pad_lastdim(tbias, "QGemm tbias", get_vector_boundary(tbias))
tbias_shift = pad_lastdim(tbias_shift, "QGemm tbias_shift", get_vector_boundary(tbias_shift))
print("int8weights\n", tw.flatten().tolist(), "\n\n\n")
print("int8bias\n", tbias_shift.flatten().tolist(), "\n\n\n")

# stream weights
tw_stream = tw.reshape(K, -1, 16).transpose(1,0,2)
print("int8weights_stream\n", tw_stream.flatten().tolist(), "\n\n\n")

tout = torch.nn.functional.linear(
  torch.Tensor(tin.astype(int) - tin_zero),
  torch.Tensor(tw.T.astype(int) - tw_zero), #NxK
  torch.Tensor(tbias[:])).numpy() * tin_scale*tw_scale/tout_scale
tout = round_away(tout) + tout_zero
tout = np.clip(tout, -128, 127).astype(np.int8)

# save
save_tensor("qgemm_int8in.txt", tin)
save_tensor(f"qgemm_int8out_shape{M}x{N}.txt", tout)
