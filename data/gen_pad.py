import numpy as np

PLIOW = 64
N = 28
INP_W = 28
OUT_W = 32

inp = np.arange(N*INP_W).reshape(N, INP_W)
out = np.pad(inp, ((0,0),(0,OUT_W-INP_W))).reshape(N, OUT_W)
np.savetxt("pad_int16in.txt", inp.reshape(-1,PLIOW//16), fmt="%d")
np.savetxt("pad_int16out_PadScalar.txt", out.reshape(-1,PLIOW//16), fmt="%d")
np.savetxt("pad_int16out_PadVector.txt", out.reshape(-1,PLIOW//16), fmt="%d")

inp = np.arange(N*INP_W).reshape(N, INP_W).astype(np.int8)
out = np.pad(inp, ((0,0),(0,OUT_W-INP_W))).reshape(N, OUT_W)
np.savetxt("pad_int8in.txt", inp.reshape(-1,PLIOW//8), fmt="%d")
np.savetxt("pad_int8out_PadScalar.txt", out.reshape(-1,PLIOW//8), fmt="%d")
np.savetxt("pad_int8out_PadVector.txt", out.reshape(-1,PLIOW//8), fmt="%d")
