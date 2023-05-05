import numpy as np
np.random.seed(0)

M = 1
K = 84
N = 10
inp = np.arange(M*K).reshape(-1, 2)
weights = np.arange(K*N).flatten().tolist()
bias = np.ones(N).flatten().tolist()
np.savetxt("gemm_fpin.txt", inp)

inp_rand = np.random.random(M*K).reshape(-1, 2)
weights_rand = np.random.random(K*N).flatten().tolist()
bias_rand = np.random.random(N).flatten().tolist()
np.savetxt("gemm_fpin_rand.txt", inp_rand)
