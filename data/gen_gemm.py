import numpy as np
np.random.seed(0)

M = 1
K = 86
N = 10
PAD = (4 - N%4) % 4

inp = np.arange(M*K).reshape(-1, 2)

weights_mknk = np.arange(K*N).reshape(K, N)
weights_mkkn = np.pad(weights_mknk, ((0,0),(0,PAD)), "constant", constant_values=0)
weights_mknk = weights_mknk.flatten().tolist()
weights_mkkn = weights_mkkn.flatten().tolist()

bias = np.ones(N).flatten().tolist()
np.savetxt("gemm_fpin.txt", inp)

inp_rand = np.random.random(M*K).reshape(-1, 2)
weights_mknk_rand = np.random.random(K*N).reshape(K, N)
weights_mkkn_rand = np.pad(weights_mknk_rand, ((0,0),(0,PAD)), "constant", constant_values=0)
weights_mknk_rand = weights_mknk_rand.flatten().tolist()
weights_mkkn_rand = weights_mkkn_rand.flatten().tolist()

bias_rand = np.random.random(N).flatten().tolist()
np.savetxt("gemm_fpin_rand.txt", inp_rand)
