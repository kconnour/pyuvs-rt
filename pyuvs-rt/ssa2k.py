import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import minimize

iteration: int = 7

results = np.genfromtxt(f'/home/kyle/ssa_retrievals/iteration{iteration}/results.csv', delimiter=',')  # shape: (19, 5)
results0 = np.genfromtxt(f'/home/kyle/ssa_retrievals/iteration{6}/results.csv', delimiter=',')  # shape: (19, 5)

lut_loc = '/home/kyle/dustssa/kyle_iuvs_2/'

cext_lut = np.load(f'{lut_loc}cext_lut.npy')
csca_lut = np.load(f'{lut_loc}csca_lut.npy')  # shape: (3, 4, 8)
ssa_lut = csca_lut / cext_lut

lut_wavs = np.array([230, 260, 300])
lut_reff = np.array([1.4, 1.6, 1.8, 2])
lut_k = np.array([1, 10, 30, 60, 100, 200, 300, 500]) / 10000


# 1. Turn SSA into k
k_spectra = np.zeros((4, 19))


def fit_k(k_guess, interp, wavelength, retr_ssa):
    if not 1/10000 <= k_guess <= 500/10000:
        return 9999999
    return (interp(wavelength, k_guess) - retr_ssa)**2


for reff in range(4):
    interpp = interp2d(lut_wavs, lut_k, ssa_lut[:, reff, :].T)  # ssa = f(wavelength, k)

    # Invert: knowing ssa and wavelength, get k
    for wav_index in range(19):
        m = minimize(fit_k, np.array([0.015]), args=(interpp, results[wav_index, 0], results[wav_index, reff+1]), method='Nelder-Mead', tol=1e-8).x[0]

        k_spectra[reff, wav_index] = m


foo = np.zeros((19, 5))
tmp = k_spectra.T
foo[:, 1:] = tmp
w = np.linspace(205, 306, num=19)
foo[:, 0] = w
np.savetxt(f'/home/kyle/ssa_retrievals/iteration{iteration}/k.csv', foo, delimiter=',')

import matplotlib.pyplot as plt
plt.plot(w, foo[:, 1])
plt.plot(w, foo[:, 2])
plt.plot(w, foo[:, 3])
plt.plot(w, foo[:, 4])
plt.savefig(f'/home/kyle/ssa_retrievals/iteration{iteration}/k.png')
