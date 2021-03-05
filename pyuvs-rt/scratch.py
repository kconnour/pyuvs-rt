# The pyuvs install command
# /home/kyle/repos/pyuvs-rt/venv/bin/python -m pip install .
from pyuvs.files import FileFinder
from pyuvs.data_contents import IUVSDataContents, L1bDataContents
from pyuvs.geography import Geography


'''g = Geography()
ff = FileFinder('/media/kyle/Samsung_T5/IUVS_data')
files = ff.soschob(7287)
for i in files.abs_paths:
    l1bc = L1bDataContents(i)
    print(g.locations['gale_crater'])
    print(g.location_in_file(l1bc, g.locations['gale_crater']))'''


#import numpy as np
#a = np.load('/home/kyle/retrieved_ssa.npy')
#print(a.shape)
#print(a[[50, 51, 52, 53, 54, 55, 56], 50, :])
#print(a[50, [50, 51, 52, 53, 54, 55, 56], :])

'''from astropy.io import fits
import numpy as np
from ssa_retrieval.gale_crater import OpticalDepth
od = OpticalDepth()
hdul = fits.open('/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels.fits')
sza = hdul['sza'].data
ea = hdul['ea'].data
ls = np.ravel(hdul['ls'].data)
pix = sza[np.where((sza <= 72) & (ea <= 72) & (od.interpolate_tau(ls) >= 5))]
print(pix.shape)'''

import numpy as np
import matplotlib.pyplot as plt

a = np.load('/home/kyle/new_retrieved_ssa.npy')
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
ref = np.nanmin(a[:, :], axis=0)

plt.plot(wavs, ref)
plt.ylim(0.6, 0.75)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Retrieved SSA')
plt.savefig('/home/kyle/ssa.png', dpi=300)
