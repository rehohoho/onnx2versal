import numpy as np
import torch

np.random.seed(0)

C = 2
W = 28
M = 2
K = 5
PAD = (8 - K%8) % 8

# arange
inp = np.arange(C*W*W)
fpweights = np.arange(M*C*K*K).reshape(M,C,K,K)
fpweights_pad = np.pad(fpweights, ((0,0),(0,0),(0,0),(0,PAD)), 
                       "constant", constant_values=0)
fpbias = np.ones((M))
np.savetxt("conv_fpin.txt", inp.reshape(-1, 2))
np.savetxt("conv_fpweights.txt", fpweights.reshape(-1, 2))
np.savetxt("conv_fpbias.txt", fpbias.reshape(-1, 2))
print("fpweights\n", fpweights.flatten().tolist(), "\n\n\n")
print("fpweights_pad\n", fpweights_pad.flatten().tolist(), "\n\n\n")
print("fpbias\n", fpbias.flatten().tolist(), "\n\n\n")

# random
inp_rand = np.random.random(C*W*W)
fpweights_rand = np.random.random(M*C*K*K).reshape(M,C,K,K)
fpweights_rand_pad = np.pad(fpweights_rand, ((0,0),(0,0),(0,0),(0,PAD)), 
                            "constant", constant_values=0)
fpbias_rand = np.random.random((M))
np.savetxt("conv_fpin_rand.txt", inp_rand.reshape(-1, 2))
np.savetxt("conv_fpweights_rand.txt", fpweights_rand.reshape(-1, 2))
np.savetxt("conv_fpbias_rand.txt", fpbias_rand.reshape(-1, 2))
print("fpweights_rand\n", fpweights_rand.flatten().tolist(), "\n\n\n")
print("fpweights_rand_pad\n", fpweights_rand_pad.flatten().tolist(), "\n\n\n")
print("fpbias_rand\n", fpbias_rand.flatten().tolist(), "\n\n\n")

# result for bchw
res_bchw = torch.nn.functional.conv2d(
  torch.Tensor(inp.reshape(1,C,W,W)), 
  torch.Tensor(fpweights.reshape(M,C,K,K)), 
  torch.Tensor(fpbias.reshape(M))).numpy()
res_bchw_rand = torch.nn.functional.conv2d(
  torch.Tensor(inp_rand.reshape(1,C,W,W)), 
  torch.Tensor(fpweights_rand.reshape(M,C,K,K)), 
  torch.Tensor(fpbias_rand.reshape(M))).numpy()
np.savetxt("conv_fpout_ConvReluScalarBCHW.txt", res_bchw.reshape(-1, 2))
np.savetxt("conv_fpout_ConvReluScalarBCHW_rand.txt", res_bchw_rand.reshape(-1, 2))
np.savetxt("conv_fpout_Conv5x5ReluBCHW.txt", res_bchw.reshape(-1, 2))
np.savetxt("conv_fpout_Conv5x5ReluBCHW_rand.txt", res_bchw_rand.reshape(-1, 2))

# result for bhwc
res_bhwc = torch.nn.functional.conv2d(
  torch.Tensor(inp.reshape(1,W,W,C).transpose(0,3,1,2)), 
  torch.Tensor(fpweights.reshape(M,K,K,C).transpose(0,3,1,2)), 
  torch.Tensor(fpbias.reshape(M))).numpy().transpose(0,2,3,1)
res_bhwc_rand = torch.nn.functional.conv2d(
  torch.Tensor(inp_rand.reshape(1,W,W,C).transpose(0,3,1,2)), 
  torch.Tensor(fpweights_rand.reshape(M,K,K,C).transpose(0,3,1,2)), 
  torch.Tensor(fpbias_rand.reshape(M))).numpy().transpose(0,2,3,1)
np.savetxt("conv_fpout_ConvReluScalarBHWC.txt", res_bhwc.reshape(-1, 2))
np.savetxt("conv_fpout_ConvReluScalarBHWC_rand.txt", res_bhwc_rand.reshape(-1, 2))
np.savetxt("conv_fpout_ConvReluScalarGmemParamBHWC.txt", res_bhwc.reshape(-1, 2))
np.savetxt("conv_fpout_ConvReluScalarGmemParamBHWC_rand.txt", res_bhwc_rand.reshape(-1, 2))
