import numpy as np
np.random.seed(0)

M = 1
K = 86
N = 10
PAD = (4 - N%4) % 4

# arange input, weights, bias
inp = np.arange(M*K)
weights_mknk = np.arange(K*N)
weights_mkkn = np.pad(weights_mknk.reshape(K, N), 
                      ((0,0),(0,PAD)), "constant", constant_values=0)
bias = np.ones(N)

np.savetxt("gemm_fpin.txt", inp.reshape(-1, 2))
np.savetxt("gemm_weights_mknk.txt", weights_mknk.reshape(-1, 2))
np.savetxt("gemm_bias.txt", bias.reshape(-1, 2))
print("weights_mknk\n", weights_mknk.flatten().tolist(), "\n\n\n")
print("weights_mkkn\n", weights_mkkn.flatten().tolist(), "\n\n\n")
print("bias\n", bias.flatten().tolist(), "\n\n\n")

# random input, weights, bias
inp_rand = np.random.random(M*K)
weights_mknk_rand = np.random.random(K*N)
weights_mkkn_rand = np.pad(weights_mknk_rand.reshape(K, N), 
                           ((0,0),(0,PAD)), "constant", constant_values=0)
bias_rand = np.random.random(N)

np.savetxt("gemm_fpin_rand.txt", inp_rand.reshape(-1, 2))
np.savetxt("gemm_weights_mknk_rand.txt", weights_mknk_rand.reshape(-1, 2))
np.savetxt("gemm_bias_rand.txt", bias_rand.reshape(-1, 2))
print("weights_mknk_rand\n", weights_mknk_rand.flatten().tolist(), "\n\n\n")
print("weights_mkkn_rand\n", weights_mkkn_rand.flatten().tolist(), "\n\n\n")
print("bias_rand\n", bias_rand.flatten().tolist(), "\n\n\n")

# result for arange
res_mknk = np.matmul(inp.reshape(1,86), weights_mknk.reshape(10,86).T) + bias
res_mknk_rand = np.matmul(inp_rand.reshape(1,86), weights_mknk_rand.reshape(10,86).T) + bias_rand
np.savetxt("gemm_fpout_GemmReluScalarMKNK.txt", res_mknk.reshape(-1, 2))
np.savetxt("gemm_fpout_GemmReluScalarMKNK_rand.txt", res_mknk_rand.reshape(-1, 2))
np.savetxt("gemm_fpout_GemmReluScalarGmemParamMKNK.txt", res_mknk.reshape(-1, 2))
np.savetxt("gemm_fpout_GemmReluScalarGmemParamMKNK_rand.txt", res_mknk_rand.reshape(-1, 2))

# result for random
res_mkkn = np.matmul(inp.reshape(1,86), weights_mknk.reshape(86,10)) + bias
res_mkkn_rand = np.matmul(inp_rand.reshape(1,86), weights_mknk_rand.reshape(86,10)) + bias_rand
np.savetxt("gemm_fpout_GemmReluScalarMKKN.txt", res_mkkn.reshape(-1, 2))
np.savetxt("gemm_fpout_GemmReluScalarMKKN_rand.txt", res_mkkn_rand.reshape(-1, 2))
np.savetxt("gemm_fpout_GemmReluMKKN.txt", res_mkkn.reshape(-1, 2))
np.savetxt("gemm_fpout_GemmReluMKKN_rand.txt", res_mkkn_rand.reshape(-1, 2))
