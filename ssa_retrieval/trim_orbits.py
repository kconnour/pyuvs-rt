from pyuvs.geography import Geography
from pyuvs.files import FileFinder
from pyuvs.data_contents import L1bDataContents
import numpy as np
from astropy.io import fits
from ssa_retrieval.my34_tau import Curiosity


class GaleCraterFinder:
    def __init__(self, data_location):
        self.__loc = data_location
        self.__lat = -5.4    # degrees N
        self.__lon = 137.8   # degrees E
        self.__radius = 77   # km

    def get_pixels(self):
        ff = FileFinder(self.__loc)
        file = []
        orbits = []
        positions = []
        integrations = []
        ls = []
        lats = []
        lons = []
        sza = []
        ea = []
        pa = []
        lt = []
        for orbit in range(7200, 7400):
            try:
                files = ff.soschob(orbit)
            except ValueError:
                continue
            for counter, path in enumerate(files.abs_paths):
                print(path)
                l1b = L1bDataContents(path)
                dist = Geography().haversine_distance(l1b.latitude[:, :, 4], l1b.longitude[:, :, 4], self.__lat, self.__lon)
                crater_pixel_indices = np.argwhere(dist < self.__radius)
                if crater_pixel_indices.size:
                    position = crater_pixel_indices[:, 0]
                    integration = crater_pixel_indices[:, 1]
                    for foo in range(len(position)):
                        p = position[foo]
                        i = integration[foo]
                        file.append(counter)
                        orbits.append(orbit)
                        positions.append(p)
                        integrations.append(i)
                        ls.append(l1b.solar_longitude)
                        lats.append(l1b.latitude[p, i, 4])
                        lons.append(l1b.longitude[p, i, 4])
                        sza.append(l1b.solar_zenith_angle[p, i])
                        ea.append(l1b.emission_angle[p, i])
                        pa.append(l1b.phase_angle[p, i])
                        lt.append(l1b.local_time[p, i])

        columns = []
        hdu = fits.PrimaryHDU()
        hdu.data = np.array([])
        columns.append(hdu)

        image = fits.ImageHDU(name='files')
        image.data = np.array(file)
        columns.append(image)

        image = fits.ImageHDU(name='orbits')
        image.data = np.array(orbits)
        columns.append(image)

        image = fits.ImageHDU(name='positions')
        image.data = np.array(positions)
        columns.append(image)

        image = fits.ImageHDU(name='integrations')
        image.data = np.array(integrations)
        columns.append(image)

        image = fits.ImageHDU(name='Ls')
        image.data = np.array(ls)
        columns.append(image)

        image = fits.ImageHDU(name='latitude')
        image.data = np.array(lats)
        columns.append(image)

        image = fits.ImageHDU(name='longitude')
        image.data = np.array(lons)
        columns.append(image)

        image = fits.ImageHDU(name='sza')
        image.data = np.array(sza)
        columns.append(image)

        image = fits.ImageHDU(name='ea')
        image.data = np.array(ea)
        columns.append(image)

        image = fits.ImageHDU(name='pa')
        image.data = np.array(pa)
        columns.append(image)

        image = fits.ImageHDU(name='lt')
        image.data = np.array(lt)
        columns.append(image)

        combined_fits = fits.HDUList(columns)
        combined_fits.writeto('/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels.fits', overwrite=True)


#gcf = GaleCraterFinder('/media/kyle/Samsung_T5/IUVS_data')
#gcf.get_pixels()

hdul = fits.open('/home/kyle/repos/pyuvs-rt/ssa_files/gale_pixels.fits')
i = hdul['integrations'].data
p = hdul['positions'].data
la = hdul['latitude'].data
lo = hdul['longitude'].data
sza = hdul['sza'].data
ea = hdul['ea'].data
ls = hdul['ls'].data

c = Curiosity()
tau = c.interpolate_tau(ls)

good = 0
for foo in range(len(i)):
    if sza[foo] <= 72 and ea[foo] <= 72 and tau[foo] >= 5:
        print(tau[foo])
        good += 1

print(good)
