import numpy as np
np.random.seed(0)
np.savetxt("transpose_in.txt", np.arange(4*4*16).reshape(-1, 2))
np.savetxt("transpose_in_rand.txt", np.random.random(4*4*16).reshape(-1, 2))
