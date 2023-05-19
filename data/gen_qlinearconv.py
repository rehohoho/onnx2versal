import numpy as np
import torch

np.random.seed(0)

INP_W = 28
OUT_W = 28
M = 8
C = 1 # loop dependency missed issue occurs at C=1
K = 1
PAD = (8 - K%8) % 8

# arange (1x1 convolution, i.e. multiplication only)
X_scale = 0.00369204697
W_scale = 0.003
Y_scale = 0.00162681262
X_zero_point = 25
W_zero_point = 0
Y_zero_point = 19
X = np.arange(C*INP_W*INP_W).astype(np.int8)
W = np.repeat([127], M*C*K*K).astype(np.int8)
B = (np.ones((M))/ X_scale / W_scale * Y_scale).astype(np.int32) # quantized, scale=xscale*wscale, zero=0
Y = Y_zero_point + X_scale*W_scale/Y_scale * (X.astype(int)-X_zero_point) * (W-W_zero_point)[0]
Y = np.tile(Y,M) + np.repeat(B,OUT_W*OUT_W) * X_scale*W_scale/Y_scale # apply bias
Y = np.round(Y)

# Testing shifting
# import ipdb;ipdb.set_trace()
# scale = X_scale*W_scale/Y_scale
# shift = int(math.log(1/scale, 2))
# off_scale = scale * 2**shift
# o1 = (y1 >> shift) * off_scale
# o2 = y1 * scale
# o1 = o1.astype(np.int32)
# o2 = o2.astype(np.int32)
# np.savetxt("o1.txt", o1.reshape(-1, 2), fmt="%f")
# np.savetxt("o2.txt", o2.reshape(-1, 2), fmt="%f")
# np.testing.assert_allclose(o1, o2, atol=0.5)

np.savetxt("qlinearconv_int8in.txt", X.reshape(-1, 8), fmt="%d")
np.savetxt("qlinearconv_int8out_QLinearConvScalar.txt", Y.astype(np.int8).reshape(-1, 8), fmt="%d")
print("int8weights\n", W.flatten().tolist(), "\n\n\n")
print("int8bias\n", B.flatten().tolist(), "\n\n\n")
