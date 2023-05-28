import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary

np.random.seed(0)


CHUNK_CNT = 10
CHUNK_SIZE = 10

tin = np.random.random(CHUNK_CNT*CHUNK_SIZE).reshape(-1, 10).astype(np.float32)
tin = pad_lastdim(tin, "tin", get_vector_boundary(tin))
np.savetxt("argmax_fpin.txt", tin.reshape(-1, 2))

out = tin.argmax(1)
np.savetxt("argmax_fpout_ArgmaxScalar_shape1x10.txt", out.reshape(-1, 2))
