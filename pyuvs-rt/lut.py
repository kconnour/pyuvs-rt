import numpy as np
import glob

# kyle_iuvs_1 was missing the 230 nm run
# kyle_iuvs_2 has 230, 260, and 300 nm
file_dir = '/home/kyle/dustssa/kyle_iuvs_2'
files = sorted(glob.glob(f'{file_dir}/*'))

# Create the LUT
shape = (3, 4, 8)
cext = np.zeros(shape)
csca = np.zeros(shape)
z11 = np.zeros((181,) + shape)

# fill the LUT
for counter, f in enumerate(files):
    z = np.genfromtxt(f, skip_header=13)[:, 1]
    ext = np.genfromtxt(f, skip_header=7, skip_footer=194 - 9, comments='D')
    sca = np.genfromtxt(f, skip_header=8, skip_footer=194 - 10, comments='D')

    z11[:, counter // 32, (counter // 8) % 4, counter % 8] = z
    cext[counter // 32, (counter // 8) % 4, counter % 8] = ext
    csca[counter // 32, (counter // 8) % 4, counter % 8] = sca

np.save('/home/kyle/dustssa/kyle_iuvs_2/z11_lut.npy', z11)
np.save('/home/kyle/dustssa/kyle_iuvs_2/cext_lut.npy', cext)
np.save('/home/kyle/dustssa/kyle_iuvs_2/csca_lut.npy', csca)
