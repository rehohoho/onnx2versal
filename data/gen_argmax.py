import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


CHUNK_CNT = 10
CHUNK_SIZE = 10

tin = np.random.random(CHUNK_CNT*CHUNK_SIZE).reshape(-1, 10).astype(np.float32)
out = tin.argmax(1).astype(np.float32)

save_tensor("argmax_fpin.txt", tin)
save_tensor("argmax_fpout_shape1x10.txt", out)
