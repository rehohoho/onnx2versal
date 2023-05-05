import numpy as np
np.random.seed(0)

C = 6
W = 24
np.savetxt("pool_in.txt", np.arange(C*W*W).reshape(-1, 2))
np.savetxt("pool_in_rand.txt", np.random.random(C*W*W).reshape(-1, 2))
