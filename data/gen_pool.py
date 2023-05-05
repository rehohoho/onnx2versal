import numpy as np
np.random.seed(0)
np.savetxt("pool_in.txt", np.arange(6*24*24).reshape(-1, 2))
np.savetxt("pool_in_rand.txt", np.random.random(6*24*24).reshape(-1, 2))
