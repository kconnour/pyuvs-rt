import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import minimize
from pyRT_DISORT.untested_utils.utilities.fit_phase_function import PhaseFunction
import matplotlib.pyplot as plt

# 0. Load files
lut_loc = '/home/kyle/dustssa/kyle_iuvs_2/'

cext_lut = np.load(f'{lut_loc}cext_lut.npy')
csca_lut = np.load(f'{lut_loc}csca_lut.npy')  # shape: (3, 4, 8)
z11_lut = np.load(f'{lut_loc}z11_lut.npy')  # shape: (181, 3, 4, 8)
ssa_lut = csca_lut / cext_lut

lut_wavs = np.array([230, 260, 300])
lut_reff = np.array([1.4, 1.6, 1.8, 2])
lut_k = np.array([1, 10, 30, 60, 100, 200, 300, 500])

retrieval15 = np.genfromtxt('/home/kyle/dustssa/1-5microns.csv', delimiter=',')
retrieval20 = np.genfromtxt('/home/kyle/dustssa/2-0microns.csv', delimiter=',')


# 1. Turn SSA into k
k_spectra = np.zeros((4, 19))


def fit_k(k_guess, wavelength, retr_ssa):
    return (interp(wavelength, k_guess) - retr_ssa)**2


for reff in range(lut_reff.shape[0]):
    interp = interp2d(lut_wavs, lut_k, ssa_lut[:, reff, :].T)  # ssa = f(wavelength, k)

    # Invert: knowing ssa and wavelength, get k
    for i in range(retrieval15[:, 0].shape[0]):
        if reff < 2:
            m = minimize(fit_k, 150, args=(retrieval15[i, 0], retrieval15[i, 1])).x[0]
            #m = np.interp()
        else:
            m = minimize(fit_k, 150, args=(retrieval20[i, 0], retrieval20[i, 1])).x[0]
        k_spectra[reff, i] = m

print(k_spectra[0, :])
print(k_spectra[1, :])
print(k_spectra[2, :])
print(k_spectra[3, :])

# 2. Get c_sca, c_ext, and z11 at k
new_csca = np.zeros((4, 19))
new_cext = np.zeros((4, 19))

for reff in range(lut_reff.shape[0]):
    f = interp2d(lut_wavs, lut_k, csca_lut[:, reff, :].T)
    g = interp2d(lut_wavs, lut_k, cext_lut[:, reff, :].T)
    for w in range(19):
        new_csca[reff, w] = f(retrieval15[w, 0], k_spectra[reff, w])[0]
        new_cext[reff, w] = g(retrieval15[w, 0], k_spectra[reff, w])[0]

# Get z11 at k
new_z11 = np.zeros((181, 4, 19))

for ang in range(181):
    for reff in range(lut_reff.shape[0]):
        f = interp2d(lut_wavs, lut_k, z11_lut[ang, :, reff, :].T)
        for w in range(19):
            new_z11[ang, reff, w] = f(retrieval15[w, 0], k_spectra[reff, w])[0]

# 2.5 normalize z11
avg = np.mean(new_z11, axis=0)

# 3. Make P(theta)
p = 4 * np.pi * new_z11 / new_csca / avg

# 4. Fit P(theta)
new_p = np.zeros((65, 4, 19))

for reff in range(4):
    for w in range(19):
        pf = PhaseFunction(p[:, reff, w], np.radians(np.linspace(0, 180, num=181)))
        a = pf.create_legendre_coefficients(65, 360)
        new_p[:, reff, w] = a

print(new_p[:, 0, 0])

#np.save('/home/kyle/dustssa/kyle_iuvs_2/new_cext.npy', new_cext)
#np.save('/home/kyle/dustssa/kyle_iuvs_2/new_csca.npy', new_csca)
#np.save('/home/kyle/dustssa/kyle_iuvs_2/new_phase.npy', new_p)
