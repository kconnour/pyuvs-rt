import os
import glob
import numpy as np
import linecache

file_dir = '/home/kyle/dustssa/kyle_iter2_radprop'

# Make phase function coeff
phsfn_files = sorted(glob.glob(f'{file_dir}/*.dat.coef'))
phsfn = np.zeros((128, 4, 4))

for counter, file in enumerate(phsfn_files):
    a = np.genfromtxt(file, skip_header=1, skip_footer=1)
    b = np.genfromtxt(file, skip_header=22)
    a = a.reshape((126,))

    wav_ind = counter % 4
    reff_ind = counter // 4
    phsfn[:126, reff_ind, wav_ind] = a
    phsfn[126:, reff_ind, wav_ind] = b


# Make forward scattering properties
fsp_files = sorted(glob.glob(f'{file_dir}/*0.dat'))
cext = np.zeros((4, 4))
csca = np.zeros((4, 4))

for counter, file in enumerate(fsp_files):
    cext_line = linecache.getline(file, 4)
    csca_line = linecache.getline(file, 5)
    # print(cext_line, file)

    wav_ind = counter % 4
    reff_ind = counter // 4
    cext[reff_ind, wav_ind] = float(cext_line[1:6])
    csca[reff_ind, wav_ind] = float(csca_line[1:6])

np.save('/home/kyle/dustssa/kyle_iter2_radprop/cext_lut.npy', cext)
np.save('/home/kyle/dustssa/kyle_iter2_radprop/csca_lut.npy', csca)
np.save('/home/kyle/dustssa/kyle_iter2_radprop/phsfn_coeff.npy', phsfn)
