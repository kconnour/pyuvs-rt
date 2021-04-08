import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from ssa_retrieval.gale_crater import OpticalDepth

# Load the retrieval
wavs = np.load('/home/kyle/iuvs_wavelengths.npy')
f = np.load('/home/kyle/ssa_retrievals/retrieved_ssa_1-5_const_prop.npy')

# Get the Gale crater pixel data
file = '/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels_slit.fits'
hdul = fits.open(file)

reflectance = hdul['reflectance'].data
position = hdul['position'].data
szas = hdul['sza'].data
eas = hdul['ea'].data
ls = hdul['ls'].data

# Make an OpticalDepth object
od = OpticalDepth()

ff = np.genfromtxt('/home/kyle/Downloads/iuvs_flatfield_133x19.txt')
rat = ff[-6, :] / ff[6, :]

inds = np.where((szas <= 72) & (eas <= 72) & (od.interpolate_tau(ls) > 5) & (f[:, 0] > 0))
plt.plot(wavs, np.mean(reflectance[inds], axis=0)*rat)
plt.savefig('/home/kyle/ssa_retrievals/newrefRat0.png')
raise SystemExit(9)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the average SSA
#ind = np.where((szas <= 72) & (eas <= 72) & (od.interpolate_tau(ls) >= 5) & (f[:, 0] > 0))
#print(np.nanmean(f[ind], axis=0))

# Plot as a function of angle
'''sza_ang = np.array([1, 2, 3, 4, 5, 6, 7, 7.2]) * 10
ea_ang = np.array([1, 2, 3, 4, 5, 6, 7, 7.2]) * 10
for sza in sza_ang:
    for ea in ea_ang:
        inds = np.where((sza - 10 <= szas) & (szas <= sza) & (ea - 10 <= eas) & (eas <= ea) & (od.interpolate_tau(ls) >= 5) & (f[:, 0] > 0))
        rfl = f[inds]
        if rfl.shape[0] == 0:
            continue
        rfl = np.mean(rfl, axis=0)
        print(rfl)
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
plt.savefig('/home/kyle/ssa_retrievals/ssa_ang-15phasefunctionhack0.png', dpi=300)'''

# Investigate the slit position
for i in range(10):
    slitrfl = np.where((szas <= 72) & (eas <= 72) & (position >= i*10) & (position < ((i+1)*10)) & (f[:, 0] > 0) & (od.interpolate_tau(ls) >= 5))
    plt.plot(wavs, np.nanmean(f[slitrfl], axis=0), label=f'{i*10}')

plt.ylim(0, 1)
plt.legend()
plt.savefig('/home/kyle/meanslitssa.png', dpi=300)
raise SystemExit(9)


































rfl = np.where((sza <= 40) & (30 <= sza) & (ea <= 72))
print(reflectance[rfl][:, 0])
a = np.nanmean(reflectance[rfl], axis=0)
plt.plot(wavs, a)
plt.xlabel('Wavelength (nm)')
plt.savefig('/home/kyle/ssa_retrievals/reflectance_average.png', dpi=300)
raise SystemExit(9)

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






