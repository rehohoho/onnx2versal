import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


INP_W = 28
B = 1
C = 1 # loop dependency missed issue occurs at C=1
M = 6
K = 5
PAD = (8 - K%8) % 8

# random
tin = np.random.random(C*INP_W*INP_W).reshape(B,C,INP_W,INP_W).astype(np.float32)
fpweights = np.random.random(M*C*K*K).reshape(M,C,K,K)
fpweights_pad = np.pad(fpweights, ((0,0),(0,0),(0,0),(0,PAD)), 
                       "constant", constant_values=0)
fpbias = np.random.random((M))

# result for bchw
tout_bchw = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M))).numpy()
tout_bchw = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M))).numpy()
save_tensor("convbchw_fpout_shape1x6x24x24.txt", tout_bchw)

# result for bhwc
tout_bhwc = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,INP_W,INP_W,C).transpose(0,3,1,2)), 
  torch.Tensor(fpweights.reshape(M,K,K,C).transpose(0,3,1,2)), 
  torch.Tensor(fpbias.reshape(M))).numpy().transpose(0,2,3,1)
tout_bhwc = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,INP_W,INP_W,C).transpose(0,3,1,2)), 
  torch.Tensor(fpweights.reshape(M,K,K,C).transpose(0,3,1,2)), 
  torch.Tensor(fpbias.reshape(M))).numpy().transpose(0,2,3,1)
save_tensor("convbhwc_fpout_shape1x6x24x24.txt", tout_bhwc)

tin = pad_lastdim(tin, "conv tin", get_vector_boundary(tin))
save_tensor("conv_fpin.txt", tin)
save_tensor("conv_fpweights.txt", fpweights)
save_tensor("conv_fpbias.txt", fpbias)
print("fpweights\n", fpweights.flatten().tolist(), "\n\n\n")
print("fpweights_pad\n", fpweights_pad.flatten().tolist(), "\n\n\n")
print("fpbias\n", fpbias.flatten().tolist(), "\n\n\n")
