import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

iteration: int = 8

# Load the retrieval
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size.npy')
g = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-6size.npy')
h = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-8size.npy')
i = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_2-0size.npy')

f_hi = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-highUnc.npy')
f_lo = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-lowUnc.npy')
g_hi = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-highUnc.npy')
g_lo = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-lowUnc.npy')
h_hi = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-highUnc.npy')
h_lo = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-lowUnc.npy')
i_hi = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-highUnc.npy')
i_lo = np.load(f'/home/kyle/ssa_retrievals/iteration{iteration}/new-fsp_new-pf_hapke-wolff_1-4size-lowUnc.npy')

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
n_pix = np.sum(np.where((szas <= 50) & (eas <= 72) & (f[:, 0] >= 0), 1, 0))

f_hi = np.sqrt(np.sum((f_hi[inds] - f[inds])**2, axis=0)) / n_pix
f_lo = np.sqrt(np.sum((f[inds] - f_lo[inds])**2, axis=0)) / n_pix
g_hi = np.sqrt(np.sum((g_hi[inds] - g[inds])**2, axis=0)) / n_pix
g_lo = np.sqrt(np.sum((g[inds] - g_lo[inds])**2, axis=0)) / n_pix
h_hi = np.sqrt(np.sum((h_hi[inds] - h[inds])**2, axis=0)) / n_pix
h_lo = np.sqrt(np.sum((h[inds] - h_lo[inds])**2, axis=0)) / n_pix
i_hi = np.sqrt(np.sum((i_hi[inds] - i[inds])**2, axis=0)) / n_pix
i_lo = np.sqrt(np.sum((i[inds] - i_lo[inds])**2, axis=0)) / n_pix

f = np.mean(f[inds], axis=0)
g = np.mean(g[inds], axis=0)
h = np.mean(h[inds], axis=0)
i = np.mean(i[inds], axis=0)

#plt.plot(wavs, f0, label='previous 1.4 microns', color=list(plt.get_cmap('plasma')(0.2)))
#plt.plot(wavs, g0, label='previous 1.6 microns', color=list(plt.get_cmap('plasma')(0.4)))
#plt.plot(wavs, h0, label='previous 1.8 microns', color=list(plt.get_cmap('plasma')(0.6)))
#plt.plot(wavs, i0, label='previous 2.0 microns', color=list(plt.get_cmap('plasma')(0.8)))

plt.plot(wavs, f, label='IUVS 1.4 micron', color=list(plt.get_cmap('cividis')(0.1)))
plt.plot(wavs, g, label='IUVS 1.6 micron', color=list(plt.get_cmap('cividis')(0.35)))
plt.plot(wavs, h, label='IUVS 1.8 micron', color=list(plt.get_cmap('cividis')(0.6)))
plt.plot(wavs, i, label='IUVS 2.0 micron', color=list(plt.get_cmap('cividis')(0.85)))

plt.fill_between(wavs, f + f_hi, f - f_lo, color=list(plt.get_cmap('cividis')(0.1)), alpha=0.5)
plt.fill_between(wavs, g + g_hi, g - g_lo, color=list(plt.get_cmap('cividis')(0.35)), alpha=0.5)
plt.fill_between(wavs, h + h_hi, h - h_lo, color=list(plt.get_cmap('cividis')(0.6)), alpha=0.5)
plt.fill_between(wavs, i + i_hi, i - i_lo, color=list(plt.get_cmap('cividis')(0.85)), alpha=0.5)

plt.ylim(0.6, 0.68)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SSA')

plt.scatter([258, 320], [0.62, 0.648], label='MARCI 1.6 microns', color=list(plt.get_cmap('cividis')(0.35)))
plt.scatter([258, 320], [0.63, 0.653], label='MARCI 1.8 microns', color=list(plt.get_cmap('cividis')(0.6)))

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
