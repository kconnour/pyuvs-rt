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
from astropy.io import fits

wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_const_prop.npy')


# Get the Gale crater pixels
file = '/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels_slit.fits'
hdul = fits.open(file)
sza = hdul['sza'].data
ea = hdul['ea'].data

rfl = np.where((sza <= 50) & (ea <= 72))

a = np.nanmean(f[rfl], axis=0)
print(a)
raise SystemExit(9)

'''reflectance = hdul['reflectance'].data



rfl = np.where((sza <= 40) & (30 <= sza) & (ea <= 72))
print(reflectance[rfl][:, 0])
a = np.nanmean(reflectance[rfl], axis=0)
plt.plot(wavs, a)
plt.xlabel('Wavelength (nm)')
plt.savefig('/home/kyle/ssa_retrievals/reflectance_average.png', dpi=300)
raise SystemExit(9)'''

# 3 microns
#retrieval_3_microns_conrath_10_5 = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_3_microns_conrath_10_5.npy')
#ref_3_microns_conrath_10_5 = np.nanmin(retrieval_3_microns_conrath_10_5[:, :], axis=0)
#plt.plot(wavs, ref_3_microns_conrath_10_5, label='3 microns')

# 2.0 microns
f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_microns_conrath_10_01-phasefunctionhack.npy')
#reflectance = np.nanmin(f[:, :], axis=0)
#print(reflectance)
#plt.plot(wavs, reflectance, label='1.8 microns')

# 2 microns
#f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_2-0_microns_conrath_10_01.npy')
#reflectance = np.nanmean(f[:, :], axis=0)
#plt.plot(wavs, reflectance, label='2 microns')

# 2.5 microns
#f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_2-5_microns_conrath_10_01.npy')
#reflectance = np.nanmean(f[:, :], axis=0)
#plt.plot(wavs, reflectance, label='2.5 microns')


# Plot as a function of angle
sza_ang = np.array([1, 2, 3, 4, 5]) * 10
ea_ang = np.array([1, 2, 3, 4, 5, 6, 7]) * 10
for sza in sza_ang:
    for ea in ea_ang:
        inds = np.where((sza - 10 <= hdul['sza'].data) & (hdul['sza'].data <= sza) & (ea - 10 <= hdul['ea'].data) & (hdul['ea'].data <= ea))
        rfl = f[inds]
        if rfl.shape[0] == 0:
            continue
        rfl = np.mean(rfl, axis=0)
        plt.plot(wavs, rfl, label=f'sza {sza}, ea {ea}')


# Wolff's
plt.scatter([258, 320], [0.62, 0.648], label='MARCI 1.6 microns', color='g')
plt.scatter([258, 320], [0.63, 0.653], label='MARCI 1.8 microns', color='hotpink')
plt.legend()

plt.ylim(0.5, 0.85)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Retrieved SSA')
plt.title('1.5 micron dust')
plt.legend()
plt.savefig('/home/kyle/ssa_retrievals/ssa_ang-15phasefunctionhack.png', dpi=300)
