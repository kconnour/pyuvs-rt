import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

iteration = 5
f0 = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size.npy')
g0 = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-6size.npy')
h0 = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-8size.npy')
i0 = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_2-0size.npy')

iteration: int = 6

# Load the retrieval
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size.npy')
g = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-6size.npy')
h = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-8size.npy')
i = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_2-0size.npy')
w = [230, 260, 280, 300]

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
f0 = np.mean(f0[inds], axis=0)
g0 = np.mean(g0[inds], axis=0)
h0 = np.mean(h0[inds], axis=0)
i0 = np.mean(i0[inds], axis=0)
f = np.mean(f[inds], axis=0)
g = np.mean(g[inds], axis=0)
h = np.mean(h[inds], axis=0)
i = np.mean(i[inds], axis=0)

# TODO: fix this in upcoming iteration
fnew = (f+f0)/2
gnew = (g+g0)/2
hnew = (h+h0)/2
inew = (i+i0)/2

plt.plot(wavs, fnew, label='IUVS 1.4 micron', color=list(plt.get_cmap('cividis')(0.1)))
plt.plot(wavs, gnew, label='IUVS 1.6 micron', color=list(plt.get_cmap('cividis')(0.35)))
plt.plot(wavs, hnew, label='IUVS 1.8 micron', color=list(plt.get_cmap('cividis')(0.6)))
plt.plot(wavs, inew, label='IUVS 2.0 micron', color=list(plt.get_cmap('cividis')(0.85)))
plt.ylim(0.55, 0.7)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SSA')

plt.scatter([258, 320], [0.62, 0.648], label='MARCI 1.6 microns', color=list(plt.get_cmap('cividis')(0.35)))
plt.scatter([258, 320], [0.63, 0.653], label='MARCI 1.8 microns', color=list(plt.get_cmap('cividis')(0.6)))

plt.legend()
plt.savefig(f'/home/kyle/ssa_retrievals/iteration{iteration}/avg0.png', dpi=300)

raise SystemExit(9)

result = np.zeros((19, 5))
result[:, 0] = wavs
result[:, 1] = fnew
result[:, 2] = gnew
result[:, 3] = hnew
result[:, 4] = inew

np.savetxt(f'/home/kyle/ssa_retrievals/iteration{iteration}/results.csv', result, delimiter=',')
