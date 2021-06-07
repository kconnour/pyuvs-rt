import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

file = '/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels.fits'

hdul = fits.open(file)
ls = hdul['ls'].data
sza = hdul['sza'].data
reflectance = hdul['reflectance'].data
f = np.load('/home/kyle/ssa_retrievals/const-fsp_const-pf_lambert-clancy-+90_1-5size.npy')

inds = np.where(f[:, 0] > 0)

plt.scatter(ls[inds], sza[inds])
plt.savefig('/home/kyle/szaDist.png')

