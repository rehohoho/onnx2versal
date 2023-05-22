import numpy as np
import torch

np.random.seed(0)

INP_W = 28
M = 6
C = 1 # loop dependency missed issue occurs at C=1
K = 5

def round_away(x):
  x = np.round(x, 3) # rounds 4.499996 and 4.496 to 4.5 first
  a = np.abs(x)
  b = np.floor(a) + np.floor(2*(a%1))
  return np.sign(x)*b

X_scale = 0.004
W_scale = 0.003
Y_scale = 0.002
X_zero_point = 25
W_zero_point = 0
Y_zero_point = 19
X = np.arange(C*INP_W*INP_W).astype(np.int8)
W = np.arange(M*C*K*K).astype(np.int8)
B = (np.arange(M) / (X_scale*W_scale/Y_scale)).astype(np.int32)

X = X.reshape(1,C,INP_W,INP_W)
n_pad = (16 - INP_W%16) % 16
if n_pad != 0:
  X = np.pad(X, ((0,0),(0,0),(0,0),(0,n_pad)), "constant", constant_values=X_zero_point)

W = W.reshape(M,C,K,K)
k_pad = (16- K%16) % 16
if k_pad != 0:
  W = np.pad(W, ((0,0),(0,0),(0,0),(0,k_pad)), "constant", constant_values=0)
if K == 5:
  W = W[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]

Y = torch.nn.functional.conv2d(
  torch.Tensor(X[:,:,:,:28].astype(int) - X_zero_point),
  torch.Tensor(W[:,:,:,(4,6,8,10,12)].astype(int) - W_zero_point),
  torch.Tensor(B[:])).numpy() * X_scale*W_scale/Y_scale
Y = round_away(Y) + Y_zero_point
Y = np.clip(Y, -128, 127).astype(np.int8)

n_pad = (16 - Y.shape[-1]%16) % 16
if n_pad != 0:
  Y = np.pad(Y, ((0,0),(0,0),(0,0),(0,n_pad)), "constant", constant_values=Y_zero_point)

np.savetxt("qlinearconv_int8in.txt", X.reshape(-1, 8), fmt="%d")
np.savetxt("qlinearconv_int8out_QLinearConvScalar.txt", Y.reshape(-1, 8), fmt="%d")
np.savetxt("qlinearconv_int8out_QLinearConvVector.txt", Y.reshape(-1, 8), fmt="%d")
print("int8weights\n", W.flatten().tolist(), "\n\n\n")
print("int8bias\n", B.flatten().tolist(), "\n\n\n")
