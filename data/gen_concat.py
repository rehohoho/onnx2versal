import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary

np.random.seed(0)


LCNT = 5
CHUNK_CNT = 4
CHUNK_SIZE = 16 # %8
BLOCK_SIZE = 52 # %4

tin = np.arange(CHUNK_CNT*CHUNK_SIZE).reshape(CHUNK_CNT,CHUNK_SIZE).astype(np.float32)
tout = np.tile(tin, 5)[:, :BLOCK_SIZE]

tin = pad_lastdim(tin, "tin", get_vector_boundary(tin))
np.savetxt("concat_fpin.txt", tin.reshape(-1,2))
np.savetxt("concat_fpout_ConcatScalar_shape4x52.txt", tout.reshape(-1, 2), fmt="%.9e")
np.savetxt("concat_fpout_ConcatVector_shape4x52.txt", tout.reshape(-1, 2))
