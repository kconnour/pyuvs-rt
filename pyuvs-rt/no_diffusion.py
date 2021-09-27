import numpy as np
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt


T = np.linspace(160, 160, num=101)   # Make an isothermal atmosphere
P0 = 600  # Surface pressure (Pascals)


def numden(z):
    n0 = P0 / Boltzmann / T[-1]
    return n0 * np.exp(-z / 10)


def diffusion(z):
    n = numden(z) / 10**6    # convert to cgs
    d = 1.52*10**18 * np.sqrt(T*(1/43 + 1/30)) / n
    return d / 10**10   # Convert to km**2


alts = np.linspace(100, 0, num=101)
concentration = np.zeros((101, 3600))
concentration[0, :] = 1


# Run the "model"
D = diffusion(alts)
K = 10**np.linspace(8, 5, num=101)
nd = numden(alts)
for t in range(3600):
    print(t)
    for ind, alt in enumerate(alts):
        if ind == 100:
            continue
        if t == 0:    # At t=0, all "stuff" is just in the 0th grid point
            continue
        else:
            deriv = concentration[ind, t] - concentration[ind+1, t-1]
            J = np.abs(D[ind] * deriv)  # using - gives odd values
            concentration[ind + 1, t] = J + concentration[ind+1, t-1]  # lossless NO, so the concentration keeps increasing

print(concentration[:, -1])
plt.semilogx(concentration[:, -1] * nd, alts)
plt.show()
