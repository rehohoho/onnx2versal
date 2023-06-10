import numpy as np

from python.op_parsers import pad_lastdim, get_vector_boundary, save_tensor

np.random.seed(0)

M = 7
K = 36
N = 10
SHUFFLE_COLSIZE = 8

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
weights_mkkn_pad_reshuffled = pad_lastdim(weights_mkkn_pad, "gemm tw_pad_shuffled", SHUFFLE_COLSIZE)
weights_mkkn_pad_reshuffled = weights_mkkn_pad_reshuffled.reshape(K, -1, SHUFFLE_COLSIZE).transpose(1,0,2)

save_tensor("gemm_fpin.txt", inp)
print("weights_mknk\n", weights_mknk.flatten().tolist(), "\n\n\n")
print("weights_mkkn\n", weights_mkkn.flatten().tolist(), "\n\n\n")
print("weights_mkkn_pad\n", weights_mkkn_pad.flatten().tolist(), "\n\n\n")
print("weights_mkkn_pad_reshuffled\n", weights_mkkn_pad_reshuffled.flatten().tolist(), "\n\n\n")
print("bias\n", bias.flatten().tolist(), "\n\n\n")
