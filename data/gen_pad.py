import numpy as np

N_PER_LINE = 4
N = 28
INP_W = 28
OUT_W = 32

inp = np.arange(N*INP_W).reshape(N, INP_W)
out = np.pad(inp, ((0,0),(0,OUT_W-INP_W))).reshape(N, OUT_W)
np.savetxt("pad_int16in.txt", inp.reshape(-1,N_PER_LINE), fmt="%d")
np.savetxt("pad_int16out.txt", out.reshape(-1,N_PER_LINE), fmt="%d")
