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
asdf = np.where(f[:, 0] > 0)
print(np.median(uncertainty[asdf] / reflectance[asdf]))
raise SystemExit(9)

position = hdul['position'].data
szas = hdul['sza'].data
eas = hdul['ea'].data
ls = hdul['ls'].data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot as a function of angle
sza_ang = np.array([1, 2, 3, 4, 5]) * 10
ea_ang = np.array([1, 2, 3, 4, 5, 6, 7, 7.2]) * 10
plot_ssa = True
plot_rfl = False
for sza in sza_ang:
    for ea in ea_ang:
        inds = np.where((sza - 10 <= szas) & (szas <= sza) & (ea - 10 <= eas) & (eas <= ea) & (f[:, 0] >= 0))
        if plot_ssa:
            ssa = f[inds]
            if ssa.shape[0] == 0:
                continue
            ssa = np.mean(ssa, axis=0)
            plt.plot(wavs, ssa, label=f'sza {sza}, ea {ea}')
        if plot_rfl:
            rfl = reflectance[inds]
            if rfl.shape[0] == 0:
                continue
            rfl = np.mean(rfl, axis=0)
            plt.plot(wavs, rfl)#, label=f'sza {sza}, ea {ea}')

# Wolff's
plt.scatter([258, 320], [0.62, 0.648], label='MARCI 1.6 microns', color='g')
plt.scatter([258, 320], [0.63, 0.653], label='MARCI 1.8 microns', color='hotpink')
plt.legend()

plt.ylim(0.5, 0.8)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SSA')
plt.title('1.5 micron dust')
plt.legend()
#plt.savefig('/home/kyle/ssa_retrievals/angles_const-fsp_const-pf_hapke-wolff_2-0size.png', dpi=300)
plt.clf()

inds = np.where((szas <= 50) & (eas <= 72) & (f[:, 0] >= 0))
ssa = f[inds]
high_ssa = f_high_unc[inds]
low_ssa = f_low_unc[inds]
d = high_ssa - ssa
e = ssa - low_ssa
hi = np.sqrt(np.mean(d**2, axis=0))
med = np.mean(ssa, axis=0)
lo = np.sqrt(np.mean(e**2, axis=0))

plt.plot(wavs, med)
plt.fill_between(wavs, med - lo, med + hi, alpha=0.2)
#plt.savefig('/home/kyle/ssa_retrievals/angles_const-fsp_const-pf_hapke-wolff_2-0size_uncertainty.png', dpi=300)

arr = np.vstack((wavs, med)).T
np.savetxt('/home/kyle/ssa_retrievals/1-5microns.csv', arr, delimiter=",")
