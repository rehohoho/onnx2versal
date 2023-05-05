import numpy as np
np.random.seed(0)

C = 16
W = 4
np.savetxt("transpose_in.txt", np.arange(C*W*W).reshape(-1, 2))
np.savetxt("transpose_in_rand.txt", np.random.random(C*W*W).reshape(-1, 2))
