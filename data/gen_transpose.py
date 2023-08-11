import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor, get_shape_str

np.random.seed(0)

B = 1
H = 4
W = 4
C = 16

bchw = np.random.random(size=(B,H,W,C)).astype(np.float32)
bhwc = bchw.transpose(0,3,1,2)

save_tensor(f"transpose_fp_bchw_{get_shape_str(bchw)}.txt", bchw)
save_tensor(f"transpose_fp_bhwc_{get_shape_str(bhwc)}.txt", bhwc)
