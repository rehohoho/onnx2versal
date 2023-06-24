import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)


H = 4
INP_W = 32 # %16 for int8, %4 for float

fpin = np.arange(H*INP_W).reshape(H,INP_W).astype(np.float32)
fpin = pad_lastdim(fpin, "concat fpin", get_vector_boundary(fpin))
save_tensor("concat_fpin.txt", fpin)

# note OUT_W %16 for int8, %4 for float
save_tensor(f"concat_fpout_shape{H}x48.txt", np.tile(fpin, 2)[:, :48])
save_tensor(f"concat_fpout_shape{H}x80.txt", np.tile(fpin, 3)[:, :80])
save_tensor(f"concat_fpout_shape{H}x112.txt", np.tile(fpin, 4)[:, :112])
save_tensor(f"concat_fpout_shape{H}x144.txt", np.tile(fpin, 5)[:, :144])
save_tensor(f"concat_fpout_shape{H}x176.txt", np.tile(fpin, 6)[:, :176])
save_tensor(f"concat_fpout_shape{H}x208.txt", np.tile(fpin, 7)[:, :208])
save_tensor(f"concat_fpout_shape{H}x240.txt", np.tile(fpin, 8)[:, :240])

int8in = fpin.astype(np.int8)
int8in = pad_lastdim(int8in, "concat int8in", get_vector_boundary(int8in))

# note OUT_W %16 for int8, %4 for float
save_tensor("concat_int8in.txt", int8in)
save_tensor(f"concat_int8out_shape{H}x48.txt", np.tile(int8in, 2)[:, :48])
save_tensor(f"concat_int8out_shape{H}x80.txt", np.tile(int8in, 3)[:, :80])
save_tensor(f"concat_int8out_shape{H}x112.txt", np.tile(int8in, 4)[:, :112])
save_tensor(f"concat_int8out_shape{H}x144.txt", np.tile(int8in, 5)[:, :144])
save_tensor(f"concat_int8out_shape{H}x176.txt", np.tile(int8in, 6)[:, :176])
save_tensor(f"concat_int8out_shape{H}x208.txt", np.tile(int8in, 7)[:, :208])
save_tensor(f"concat_int8out_shape{H}x240.txt", np.tile(int8in, 8)[:, :240])

# multi concat
save_tensor(f"concatmulti_fpout_shape{H}x816.txt", np.tile(fpin, 52)[:, :816])
