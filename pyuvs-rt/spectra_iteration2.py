import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the retrieval
# Note: you nincompoop, you forgot to modify the path in g, h, and i in retrieval 2. That's why f showed differences but the other 3 didn't
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load('/home/kyle/ssa_retrievals/iteration2-1/new-fsp_new-pf_hapke-wolff_1-4size.npy')
g = np.load('/home/kyle/ssa_retrievals/iteration2-1/new-fsp_new-pf_hapke-wolff_1-6size.npy')
h = np.load('/home/kyle/ssa_retrievals/iteration2-1/new-fsp_new-pf_hapke-wolff_1-8size.npy')
i = np.load('/home/kyle/ssa_retrievals/iteration2-1/new-fsp_new-pf_hapke-wolff_2-0size.npy')
#f_high_unc = np.load('/home/kyle/ssa_retrievals/const-fsp_const-pf_hapke-wolff_1-5size_high-uncertainty.npy')
#f_low_unc = np.load('/home/kyle/ssa_retrievals/const-fsp_const-pf_hapke-wolff_1-5size_low-uncertainty.npy')
# Note regarding f: It has shape (6857, 19). 4774 of them are a NaN (they have
#  OD < 5 or too high SZA or EA. 2083 of them fit the OD and angular criteria

# Get the Gale crater pixel data
file = '/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels_slit.fits'
hdul = fits.open(file)

reflectance = hdul['reflectance'].data
uncertainty = hdul['uncertainty'].data

position = hdul['position'].data
szas = hdul['sza'].data
eas = hdul['ea'].data
ls = hdul['ls'].data

# Plot averages
inds = np.where((szas <= 50) & (eas <= 72) & (f[:, 0] >= 0))
f = np.mean(f[inds], axis=0)
g = np.mean(g[inds], axis=0)
h = np.mean(h[inds], axis=0)
i = np.mean(i[inds], axis=0)
plt.plot(wavs, f, label='1.4 micron average')
plt.plot(wavs, g, label='1.6 micron average')
plt.plot(wavs, h, label='1.8 micron average')
plt.plot(wavs, i, label='2.0 micron average')
plt.ylim(0.5, 0.8)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SSA')

plt.scatter([258, 320], [0.62, 0.648], label='MARCI 1.6 microns', color='g')
plt.scatter([258, 320], [0.63, 0.653], label='MARCI 1.8 microns', color='hotpink')

plt.legend()
plt.savefig('/home/kyle/ssa_retrievals/iteration2-1/avg.png')

result = np.zeros((19, 5))
result[:, 0] = wavs
result[:, 1] = f
result[:, 2] = g
result[:, 3] = h
result[:, 4] = i

np.savetxt('/home/kyle/ssa_retrievals/iteration2-1/results.csv', result, delimiter=',')
