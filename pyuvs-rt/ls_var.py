import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the retrieval
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load('/home/kyle/ssa_retrievals/const-fsp_const-pf_hapke-wolff_1-5size.npy')
f_high_unc = np.load('/home/kyle/ssa_retrievals/const-fsp_const-pf_hapke-wolff_1-5size_high-uncertainty.npy')
f_low_unc = np.load('/home/kyle/ssa_retrievals/const-fsp_const-pf_hapke-wolff_1-5size_low-uncertainty.npy')
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
print(ls.shape)
inds = np.where((szas >= 50) & (szas <= 60) & (eas <= 72) & (f[:, 0] >= 0))# & (ls < 205))
ssa = f[inds]

print(ssa.shape)
for i in range(ssa.shape[0]):
    plt.plot(ssa[i])
plt.savefig('/home/kyle/ssa_retrievals/ssa_ls.png', dpi=300)
