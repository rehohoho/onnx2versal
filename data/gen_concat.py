import numpy as np
import torch

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


LCNT = 5
CHUNK_CNT = 4
CHUNK_SIZE = 16 # %8
BLOCK_SIZE = 52 # %4

tin = np.arange(CHUNK_CNT*CHUNK_SIZE).reshape(CHUNK_CNT,CHUNK_SIZE).astype(np.float32)
tout = np.tile(tin, 5)[:, :BLOCK_SIZE]

tin = pad_lastdim(tin, "tin", get_vector_boundary(tin))
save_tensor("concat_fpin.txt", tin)
save_tensor("concat_fpout_shape4x52.txt", tout)
