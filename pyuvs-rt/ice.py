"""This will make the ice scattering properties"""

import glob
import numpy as np

# Make the phase function
scat_files = sorted(glob.glob('/home/kyle/ice/*scat*'))
phase_coeff = np.zeros((13, 1568, 65))  # 26 files... one for forward scat, one for phase function
phase_psize = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80]) / 10

for counter, file in enumerate(scat_files):
    scat_arr = np.genfromtxt(file, skip_header=3)
    phase_coeff[counter, :, :] = scat_arr[:, 3:]

np.save('/home/kyle/ice/ice_phase.npy', phase_coeff)
np.save('/home/kyle/ice/ice_wavs.npy', phase_coeff[:, 0])
np.save('/home/kyle/ice/ice_psize.npy', phase_psize)

# Make the forward scattering
forw_files = sorted(glob.glob('/home/kyle/ice/*forw*'))

forw_scat = np.zeros((13, 1568, 4))

for counter, file in enumerate(forw_files):
    forw_arr = np.genfromtxt(file, skip_header=2)
    forw_scat[counter, :, :]= forw_arr

np.save('/home/kyle/ice/ice_forwardscat.npy', forw_scat)
