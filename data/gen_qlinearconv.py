import numpy as np
import torch

np.random.seed(0)

INP_W = 28
M = 8
C = 1 # loop dependency missed issue occurs at C=1
K = 1
PAD = (8 - K%8) % 8

# arange
X = np.arange(C*INP_W*INP_W).astype(np.int8)
W = np.repeat([127], M*C*K*K).astype(np.int8)
B = np.ones((M)).astype(np.int8)
X_scale = 0.00369204697
W_scale = 0.003
Y_scale = 0.00162681262
X_zero_point = np.array(25).astype(np.int8)
W_zero_point = 0
Y_zero_point = 19
Y = Y_zero_point + X_scale*W_scale/Y_scale * (X-X_zero_point) * (W-W_zero_point)[0]
Y = np.tile(Y, M)

np.savetxt("qlinearconv_int8in.txt", X.reshape(-1, 8), fmt="%d")
np.savetxt("qlinearconv_int8out_QLinearConvScalar.txt", Y.astype(np.int8).reshape(-1, 8), fmt="%d")
print("int8weights\n", W.flatten().tolist(), "\n\n\n")
print("int8bias\n", B.flatten().tolist(), "\n\n\n")
