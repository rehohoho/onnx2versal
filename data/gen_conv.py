import numpy as np

inp = np.arange(6*12*12).reshape(-1, 2)
weights = np.arange(16*6*5*5).reshape(-1, 2)
bias = np.ones((16)).reshape(-1, 2)
np.savetxt("conv_in.txt", inp)
np.savetxt("conv_weights.txt", weights)
np.savetxt("conv_bias.txt", bias)