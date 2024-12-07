import numpy as np
import matplotlib.pyplot as plt

N = np.array([500**.25, 1000**.25, 2000**.25])
err1st = np.array([0.025083555193075237, 0.01551468625517215,  0.01019021299019001])
err2nd = np.array([0.01922210175886961, 0.013682264631896653, 0.009574941601158685])

plt.loglog(N, err1st, label="1st Order Flux")
plt.loglog(N, err2nd, label="2nd Order Flux")
plt.xlabel("Grid Resolution N")
plt.ylabel("L2 Norm Error")
plt.title("L2 Norm Error vs. Grid Resolution")
plt.grid(True, which='both', linestyle='--')
plt.legend(loc='upper right')

plt.show()

initial_slope = (np.log(err1st[2] / err1st[1])) / (np.log(N[2] / N[1]))

print('The initial slope is really close to 2nd order, but then it drops! :', initial_slope)

initial_slope = (np.log(err2nd[2] / err2nd[1])) / (np.log(N[2] / N[1]))

print('The initial slope is really close to 2nd order, but then it drops! :', initial_slope)