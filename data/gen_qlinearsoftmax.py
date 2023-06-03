import numpy as np
import scipy

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor, round_away

np.random.seed(0)


INP_H = 10
INP_W = 20

X_scale = 0.004
Y_scale = 0.003
X_zero_point = -128
Y_zero_point = -128

# qtin = np.random.randint(-128, 128, size=(INP_H,INP_W)).astype(np.int8)
qtin = np.arange(-128, 128).astype(np.int8)[:INP_H*INP_W].reshape(INP_H, INP_W)
tout = scipy.special.softmax((qtin.astype(int) - X_zero_point) * X_scale, axis=1)
qtout = round_away(tout/Y_scale) + Y_zero_point
qtout = np.clip(qtout, -128, 127).astype(np.int8)

qtin = pad_lastdim(qtin, "qlinearsoftmax qtin", get_vector_boundary(qtin), value=X_zero_point)
INP_W_PAD = qtin.shape[-1]
save_tensor(f"qlinearsoftmax_int8in.txt", qtin)
save_tensor(f"qlinearsoftmax_int8out_shape{INP_H}x{INP_W}.txt", qtout)
