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


def get_vector_boundary(tensor: np.ndarray):
  return VECTOR_WORD_BOUNDARY // tensor.dtype.itemsize


INP_W = 28
M = 6
C = 1 # loop dependency missed issue occurs at C=1
K = 5
X_scale = 0.004
W_scale = 0.003
Y_scale = 0.002
X_zero_point = 25
W_zero_point = 0
Y_zero_point = 19

X = np.arange(C*INP_W*INP_W).astype(np.int8).reshape(1,C,INP_W,INP_W)
W = np.arange(M*C*K*K).astype(np.int8).reshape(M,C,K,K)
B = (np.arange(M) / (X_scale*W_scale/Y_scale)).astype(np.int32)

X = pad_lastdim(X, "X", get_vector_boundary(X))
W = pad_lastdim(W, "W", get_vector_boundary(W))
if K == 5:
  W = W[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]

Y = torch.nn.functional.conv2d(
  torch.Tensor(X[:,:,:,:28].astype(int) - X_zero_point),
  torch.Tensor(W[:,:,:,(4,6,8,10,12)].astype(int) - W_zero_point),
  torch.Tensor(B[:])).numpy() * X_scale*W_scale/Y_scale
Y = round_away(Y) + Y_zero_point
Y = np.clip(Y, -128, 127).astype(np.int8)
Y = pad_lastdim(Y, "Y", get_vector_boundary(Y))

np.savetxt("qlinearconv_int8in.txt", X.reshape(-1, 8), fmt="%d")
np.savetxt("qlinearconv_int8out_QLinearConvScalar_shape1x6x24x24.txt", Y.reshape(-1, 8), fmt="%d")
np.savetxt("qlinearconv_int8out_QLinearConvVector_shape1x6x24x24.txt", Y.reshape(-1, 8), fmt="%d")
print("int8weights\n", W.flatten().tolist(), "\n\n\n")
print("int8bias\n", B.flatten().tolist(), "\n\n\n")
