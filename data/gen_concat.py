import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


LCNT = 5
CHUNK_CNT = 4
CHUNK_SIZE = 32 # %16
BLOCK_SIZE = 112 # %16

fpin = np.arange(CHUNK_CNT*CHUNK_SIZE).reshape(CHUNK_CNT,CHUNK_SIZE).astype(np.float32)
fpout = np.tile(fpin, LCNT)[:, :BLOCK_SIZE]
int8in = fpin.astype(np.int8)
int8out = np.tile(int8in, LCNT)[:, :BLOCK_SIZE]

fpin = pad_lastdim(fpin, "fpin", get_vector_boundary(fpin))
save_tensor("concat_fpin.txt", fpin)
save_tensor("concat_fpout_shape4x112.txt", fpout)

int8in = pad_lastdim(int8in, "int8in", get_vector_boundary(int8in))
save_tensor("concat_int8in.txt", int8in)
save_tensor("concat_int8out_shape4x112.txt", int8out)

LCNT = 52
BLOCK_SIZE = 816 # %16
tout = np.tile(fpin, LCNT)[:, :BLOCK_SIZE]
save_tensor("concatmulti_fpout_shape4x816.txt", tout)
