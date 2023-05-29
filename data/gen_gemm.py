import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)

M = 1
K = 86
N = 10

# random input, weights, bias
inp = np.random.random(size=(M,K)).astype(np.float32)
weights_mknk = np.random.random(size=(N,K)).astype(np.float32)
weights_mkkn = np.random.random(size=(K,N)).astype(np.float32)
bias = np.random.random(N).astype(np.float32)

# result for mknk
res_mknk = np.matmul(inp, weights_mknk.T) + bias
save_tensor(f"gemmMKNK_fpout_shape{M}x{N}.txt", res_mknk)

# result for mkkn
res_mkkn = np.matmul(inp, weights_mkkn) + bias
save_tensor(f"gemmMKKN_fpout_shape{M}x{N}.txt", res_mkkn)

weights_mkkn_pad = pad_lastdim(weights_mkkn, "gemm tw", get_vector_boundary(weights_mkkn))
save_tensor("gemm_fpin.txt", inp)
save_tensor("gemm_fpweights_mknk.txt", weights_mknk)
save_tensor("gemm_fpbias.txt", bias)
print("weights_mknk\n", weights_mknk.flatten().tolist(), "\n\n\n")
print("weights_mkkn\n", weights_mkkn.flatten().tolist(), "\n\n\n")
print("weights_mkkn_pad\n", weights_mkkn_pad.flatten().tolist(), "\n\n\n")
print("bias\n", bias.flatten().tolist(), "\n\n\n")
