import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor


N = 28
INP_W = 28
OUT_W = 32

inp = np.arange(N*INP_W).reshape(N, INP_W).astype(np.int16)
out = np.pad(inp, ((0,0),(0,OUT_W-INP_W))).reshape(N, OUT_W)
save_tensor("pad_int16in.txt", inp)
save_tensor("pad_int16out_PadScalar_shape28x32.txt", out)
save_tensor("pad_int16out_PadVector_shape28x32.txt", out)

inp = np.arange(N*INP_W).reshape(N, INP_W).astype(np.int8)
out = np.pad(inp, ((0,0),(0,OUT_W-INP_W))).reshape(N, OUT_W)
save_tensor("pad_int8in.txt", inp)
save_tensor("pad_int8out_PadVector_shape28x32.txt", out)
