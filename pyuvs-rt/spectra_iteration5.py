import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

iteration: int = 5

# Load the retrieval
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size.npy')
g = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-6size.npy')
h = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-8size.npy')
i = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_2-0size.npy')
w = [230, 260, 280, 300]
# TODO: Fix (delete) these iteration 5
fcorr = np.interp(wavs, w, [0, -0.005, 0, 0])
gcorr = np.interp(wavs, w, [-0.03, -0.01, -0.025, -0.025])
hcorr = np.interp(wavs, w, [-0.01, -0.005, 0.01, 0.02])
icorr = np.interp(wavs, w, [-0.03, -0.03, 0, 0.01])

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
f = np.mean(f[inds], axis=0) + fcorr
g = np.mean(g[inds], axis=0)# + gcorr
h = np.mean(h[inds], axis=0)# + hcorr
i = np.mean(i[inds], axis=0)# + icorr
#plt.errorbar(wavs, f, yerr=0.01, capsize=3, label='1.4 micron average')
plt.plot(wavs, f, label='1.4 micron average')
plt.plot(wavs, g, label='1.6 micron average')
plt.plot(wavs, h, label='1.8 micron average')
plt.plot(wavs, i, label='2.0 micron average')
plt.ylim(0.55, 0.7)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SSA')

plt.scatter([258, 320], [0.62, 0.648], label='MARCI 1.6 microns', color='g')
plt.scatter([258, 320], [0.63, 0.653], label='MARCI 1.8 microns', color='hotpink')

plt.legend()
plt.savefig(f'/home/kyle/ssa_retrievals/iteration{iteration}/avg.png', dpi=300)

raise SystemExit(9)
result = np.zeros((19, 5))
result[:, 0] = wavs
result[:, 1] = f
result[:, 2] = g
result[:, 3] = h
result[:, 4] = i

np.savetxt(f'/home/kyle/ssa_retrievals/iteration{iteration}/results.csv', result, delimiter=',')
