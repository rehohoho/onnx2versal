import numpy as np
import torch

np.random.seed(0)

LCNT = 5
WINDOW_SIZE = 64
CHUNK_SIZE = 16 # %8
BLOCK_SIZE = 52 # %4

# arange
inp = np.arange(WINDOW_SIZE).reshape(-1,CHUNK_SIZE)
np.savetxt("concat_fpin.txt", inp.reshape(-1,2))

# arange result
res = np.tile(inp, 5)[:, :BLOCK_SIZE]
np.savetxt("concat_fpout_ConcatScalar.txt", res.reshape(-1, 2))
np.savetxt("concat_fpout_ConcatVector.txt", res.reshape(-1, 2))
