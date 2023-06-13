import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, get_shape_str, save_tensor

np.random.seed(0)


INP_H = 24
INP_W = 24
B = 1
C = 1 # loop dependency missed issue occurs at C=1
M = 5
K = 5
PAD = (8 - K%8) % 8

# random
tin = np.random.random(C*INP_W*INP_W).reshape(B,C,INP_W,INP_W).astype(np.float32) - 0.5
fpweights = np.random.random(M*C*K*K).reshape(M,C,K,K).astype(np.float32) - 0.5
fpweights_pad = np.pad(fpweights, ((0,0),(0,0),(0,0),(0,PAD)), 
                       "constant", constant_values=0)
fpbias = np.random.random((M)).astype(np.float32) - 0.5

# result for bchw
tout_bchw = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), padding="same").numpy()
tout_bchw = torch.nn.functional.relu(torch.Tensor(tout_bchw)).numpy()
save_tensor(f"convbchw_fpout_{get_shape_str(tout_bchw)}.txt", tout_bchw)

tout_bchw_stride2 = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), stride=2).numpy()
tout_bchw_stride2 = torch.nn.functional.relu(torch.Tensor(tout_bchw_stride2)).numpy()
save_tensor(f"convbchw_fpout_stride2_{get_shape_str(tout_bchw_stride2)}.txt", tout_bchw_stride2)

# result for bhwc
tout_bhwc = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,INP_W,INP_W,C).transpose(0,3,1,2)), 
  torch.Tensor(fpweights.reshape(M,K,K,C).transpose(0,3,1,2)), 
  torch.Tensor(fpbias.reshape(M)), padding="same").numpy().transpose(0,2,3,1)
tout_bhwc = torch.nn.functional.relu(torch.Tensor(tout_bhwc)).numpy()
save_tensor(f"convbhwc_fpout_{get_shape_str(tout_bhwc)}.txt", tout_bhwc)

tin = pad_lastdim(tin, "conv tin", get_vector_boundary(tin))
save_tensor("conv_fpin.txt", tin)
print("fpweights\n", fpweights.flatten().tolist(), "\n\n\n")
print("fpweights_pad\n", fpweights_pad.flatten().tolist(), "\n\n\n")
print("fpbias\n", fpbias.flatten().tolist(), "\n\n\n")
