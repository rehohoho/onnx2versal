import numpy as np
import scipy

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


CHUNK_CNT = 10
CHUNK_SIZE = 10

tin = np.random.random(CHUNK_CNT*CHUNK_SIZE).reshape(-1, 10).astype(np.float32)
tout = scipy.special.softmax(tin, axis=1)

tin = pad_lastdim(tin, "softmax tin", 8)
save_tensor("softmax_fpin.txt", tin)
save_tensor("softmax_fpout_shape10x10.txt", tout)
