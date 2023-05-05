import numpy as np
np.random.seed(0)

C = 6
W = 12
M = 8
K = 5
inp = np.arange(C*W*W).reshape(-1, 2)
weights = np.arange(M*C*K*K).flatten().tolist()
bias = np.ones((M)).flatten().tolist()
np.savetxt("conv_fpin.txt", inp)

inp_rand = np.random.random(C*W*W).reshape(-1, 2)
weights_rand = np.random.random(M*C*K*K).flatten().tolist()
bias_rand = np.random.random((M)).flatten().tolist()
np.savetxt("conv_fpin_rand.txt", inp_rand)
