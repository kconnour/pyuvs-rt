from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

hdul = fits.open('/media/kyle/Samsung_T5/IUVS_data/orbit07600/mvn_iuv_l1b_apoapse-orbit07642-muv_20180901T111448_v13_r01.fits.gz')

lon = hdul['pixelgeometry'].data['pixel_corner_lat']
print(lon.shape)

print(np.amin(lon[:, :, 4]))
