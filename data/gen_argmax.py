import numpy as np
np.random.seed(0)

WINDOW_SIZE = 100
CHUNK_SIZE = 10
inp = np.tile(np.arange(CHUNK_SIZE - 1), WINDOW_SIZE//(CHUNK_SIZE-1) + 1)[:WINDOW_SIZE].reshape(-1, 2)
np.savetxt("argmax_fpin.txt", inp)

inp_rand = np.random.random(WINDOW_SIZE).reshape(-1, 2)
np.savetxt("argmax_fpin_rand.txt", inp_rand)
