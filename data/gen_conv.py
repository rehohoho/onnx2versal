import numpy as np

inp = np.arange(28*28).reshape(-1, 2)
weights = np.arange(5*5)
weights = np.tile(weights, 6).reshape(-1, 2)
bias = np.ones((6)).reshape(-1, 2)
np.savetxt("conv_in.txt", inp)
np.savetxt("conv_weights.txt", weights)
np.savetxt("conv_bias.txt", bias)