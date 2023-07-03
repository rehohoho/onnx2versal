import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, get_shape_str, save_tensor

np.random.seed(0)


INP_H = 24
INP_W = 24
B = 1
C = 2 # loop dependency missed issue occurs at C=1
M = 4 # even number if padding/striding cause H and W to be non-even
K = 5

# 5x5
tin = np.random.random(C*INP_W*INP_W).reshape(B,C,INP_W,INP_W).astype(np.float32) - 0.5
save_tensor("conv_fpin.txt", pad_lastdim(tin, "conv tin", get_vector_boundary(tin)))

fpweights_5x5 = np.random.random(M*C*K*K).reshape(M,C,K,K).astype(np.float32) - 0.5
fpweights_5x5_pad = pad_lastdim(fpweights_5x5, "conv fpweights_5x5", 8)
fpbias = np.random.random((M)).astype(np.float32) - 0.5
print("fpweights_5x5\n", fpweights_5x5.flatten().tolist(), "\n\n\n")
print("fpweights_5x5_pad\n", fpweights_5x5_pad.flatten().tolist(), "\n\n\n")
print("fpbias\n", fpbias.flatten().tolist(), "\n\n\n")

# bchw
tout_bchw = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights_5x5.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), padding="same").numpy()
tout_bchw = torch.nn.functional.relu(torch.Tensor(tout_bchw)).numpy()
save_tensor(f"convbchw_fpout_{get_shape_str(tout_bchw)}.txt", tout_bchw)

tout_bchw_stride2 = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights_5x5.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), stride=2).numpy()
tout_bchw_stride2 = torch.nn.functional.relu(torch.Tensor(tout_bchw_stride2)).numpy()
save_tensor(f"convbchw_fpout_stride2_{get_shape_str(tout_bchw_stride2)}.txt", tout_bchw_stride2)


# 3x3
K = 3
fpweights_3x3 = np.random.random(M*C*K*K).reshape(M,C,K,K).astype(np.float32) - 0.5
fpweights_3x3_pad = pad_lastdim(fpweights_3x3.reshape(M,C,-1), "conv fpweights_3x3", 12)
fpweights_3x3_groupC = pad_lastdim(fpweights_3x3.reshape(M,C,-1)[:,0:1,...], "conv fpweights_3x3", 12)
print("fpweights_3x3\n", fpweights_3x3.flatten().tolist(), "\n\n\n")
print("fpweights_3x3_pad\n", fpweights_3x3_pad.flatten().tolist(), "\n\n\n")
print("fpweights_3x3_groupC\n", fpweights_3x3_groupC.flatten().tolist(), "\n\n\n")

tout_bchw_3x3 = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights_3x3.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), padding="same").numpy()
tout_bchw_3x3 = torch.nn.functional.relu(torch.Tensor(tout_bchw_3x3)).numpy()
save_tensor(f"convbchw_fpout_3x3_{get_shape_str(tout_bchw_3x3)}.txt", tout_bchw_3x3)

# At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
tout_bchw_3x3_groupC = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights_3x3.reshape(M,C,K,K)[:,0:1,...]), 
  torch.Tensor(fpbias.reshape(M)), padding="same", groups=C).numpy()
tout_bchw_3x3_groupC = torch.nn.functional.relu(torch.Tensor(tout_bchw_3x3_groupC)).numpy()
save_tensor(f"convbchw_fpout_3x3_group{C}_{get_shape_str(tout_bchw_3x3_groupC)}.txt", tout_bchw_3x3_groupC)

tout_bchw_3x3_stride2 = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights_3x3.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), stride=2).numpy()
tout_bchw_3x3_stride2 = torch.nn.functional.relu(torch.Tensor(tout_bchw_3x3_stride2)).numpy()
save_tensor(f"convbchw_fpout_3x3_stride2_{get_shape_str(tout_bchw_3x3_stride2)}.txt", tout_bchw_3x3_stride2)


# 1x1
K = 1

fpweights_1x1 = np.random.random(M*C*K*K).reshape(M,C,K,K).astype(np.float32) - 0.5
fpweights_1x1_pad = pad_lastdim(fpweights_1x1.reshape(M,-1), "conv fpweights_1x1", get_vector_boundary(fpweights_1x1))
print("fpweights_1x1\n", fpweights_1x1.flatten().tolist(), "\n\n\n")
print("fpweights_1x1_pad\n", fpweights_1x1_pad.flatten().tolist(), "\n\n\n")

tout_bchw_1x1 = torch.nn.functional.conv2d(
  torch.Tensor(tin.reshape(1,C,INP_W,INP_W)), 
  torch.Tensor(fpweights_1x1.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M)), padding="same").numpy()
tout_bchw_1x1 = torch.nn.functional.relu(torch.Tensor(tout_bchw_1x1)).numpy()
save_tensor(f"convbchw_fpout_1x1_{get_shape_str(tout_bchw_1x1)}.txt", tout_bchw_1x1)
