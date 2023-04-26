import numpy as np

inp = np.arange(24*24)
inp = np.tile(inp, 6).reshape(-1, 2)
np.savetxt("pool_in.txt", inp)