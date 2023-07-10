import numpy as np

from python.op_parsers import pad_lastdim, get_shape_str, get_vector_boundary, save_tensor
np.random.seed(0)

B = 2
INP_H = 30
INP_W = 30
INP_W_PAD = 32
H0 = 1
H1 = 1
W0 = 1
W1 = 1
OUT_H = INP_H + H0 + H1
OUT_W = INP_W + W0 + W1

fpin = np.random.random(size=(B, INP_H, INP_W)).astype(np.float32)
fpout = np.pad(fpin, ((0,0),(H0,H1),(W0,W1)))

fpin = pad_lastdim(fpin, "pad fpin", INP_W_PAD)
save_tensor(f"pad_2d_fpin_{get_shape_str(fpin)}.txt", fpin)
save_tensor(f"pad_2d_fpout_{get_shape_str(fpout)}.txt", fpout)

int8in = np.random.randint(-128, 128, size=(B, INP_H, INP_W)).astype(np.int8)
int8out = np.pad(int8in, ((0,0),(H0,H1),(W0,W1)))

int8in = pad_lastdim(int8in, "pad int8in", INP_W_PAD)
save_tensor(f"pad_2d_int8in_{get_shape_str(int8in)}.txt", int8in)
save_tensor(f"pad_2d_int8out_{get_shape_str(int8out)}.txt", int8out)
