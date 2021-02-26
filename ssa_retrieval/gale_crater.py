"""gale_crater contains classes to pull relevant info from multiple spacecraft
about Gale Crater
"""

import os
from astropy.io import fits
import numpy as np
import pandas as pd
from pyuvs.data_contents import L1bDataContents
from pyuvs.files import FileFinder, DataPath
from pyuvs.geography import Geography


class GaleCraterCoordinates:
    """Make the coordinates of Gale Crater

    """
    def __init__(self):
        self.__latitude = -5.4
        self.__longitude = 137.8
        self.__radius = 77

    @property
    def latitude(self):
        """Get the latitude [degrees N] of Gale Crater.

        Returns
        -------
        float
            The crater's center latitude.

        """
        return self.__latitude

    @property
    def longitude(self):
        """Get the longitude [degrees E] of Gale Crater.

        Returns
        -------
        float
            The crater's center longitude.

        """
        return self.__longitude

    @property
    def radius(self):
        """Get the radius [km] of Gale Crater.

        Returns
        -------
        float
            The crater's radius.

        """
        return self.__radius


class OpticalDepth:
    """Get the optical depths during the MY34 global dust storm.

    OpticalDepth reads in the optical depths as measured by Curiosity and
    can interpolate them at any solar longitude.

    """
    def __init__(self):
        self.__curiosity_tau = \
            os.path.abspath('../ssa_files/gale_crater_tau.csv')

    def interpolate_tau(self, solar_longitude):
        """Interpolate the measured optical depth to an input solar longitude.

        Parameters
        ----------
        solar_longitude: np.ndarray
            The measured solar longitudes.

        Returns
        -------
        np.ndarray
            The optical depths at the input solar longitudes.

        """
        df = pd.read_csv(self.__curiosity_tau)
        solar_longitude_grid = df['Solar Longitude'].to_numpy()
        tau_grid = df['Tau'].to_numpy()
        return np.interp(solar_longitude, solar_longitude_grid, tau_grid)


class IUVSPixels:
    """Get the pixels from IUVS data that were taken over Gale Crater

    """
    def __init__(self, data_location: str) -> None:
        self.__location = data_location
        self.__gale = GaleCraterCoordinates()

    def get_pixels_over_crater(self, orbit_start: int, orbit_end: int):
        ff = FileFinder(self.__location)
        crater_files = []
        crater_positions = []
        crater_integrations = []
        for orbit in range(orbit_start, orbit_end):
            try:
                files = ff.soschob(orbit)
            except ValueError:
                continue
            for counter, fname in enumerate(files.filenames):
                l1b = L1bDataContents(fname.path)
                dist = Geography().haversine_distance(
                    l1b.latitude[:, :, 4], l1b.longitude[:, :, 4],
                    self.__gale.latitude, self.__gale.longitude)
                crater_pixel_indices = np.argwhere(dist < self.__gale.radius)
                if crater_pixel_indices.size:
                    crater_files.append(fname.path)
                    crater_positions.append(crater_pixel_indices[:, 0])
                    crater_integrations.append(crater_pixel_indices[:, 1])
            print(orbit)
        return crater_files, crater_positions, crater_integrations

    def make_gale_fits(self, orbit_start, orbit_end, l1c_location, save_location):
        files, positions, integrations = \
            self.get_pixels_over_crater(orbit_start, orbit_end)
        reflectance = []
        uncertainty = []
        wavelengths = []
        ls = []
        lat = []
        lon = []
        sza = []
        ea = []
        pa = []

        columns = []
        hdu = fits.PrimaryHDU()
        hdu.data = np.array([])
        columns.append(hdu)

        # TODO: check the index order
        for counter, f in enumerate(files):
            l1b = L1bDataContents(f)
            print(l1b.emission_angle.shape)
            l1cpath = DataPath(l1c_location).block(int(l1b.orbit_number))
            fname = os.path.basename(f).replace('l1b', 'l1c').replace('.gz', '')
            #l1c = fits.open(os.path.join(l1cpath, fname))
            n_pixels = len(positions[counter])
            for pixel in range(n_pixels):
                #reflectance.append(l1c['reflectance'].data[integrations, positions, :])
                #uncertainty.append(l1c['uncertainty'].data[integrations, positions, :])
                wavelengths.append(l1b.wavelengths[0, positions[counter][pixel], :])
                ls.append([l1b.solar_longitude])
                lat.append(l1b.latitude[integrations[counter][pixel], positions[counter][pixel], 4])
                lon.append(l1b.longitude[integrations[counter][pixel], positions[counter][pixel], 4])
                sza.append(l1b.solar_zenith_angle[integrations[counter][pixel], positions[counter][pixel]])
                ea.append(l1b.emission_angle[integrations[counter][pixel], positions[counter][pixel]])
                pa.append(l1b.phase_angle[integrations[counter][pixel], positions[counter][pixel]])

        wavelengths = np.array(wavelengths)
        ls = np.array(ls)
        lat = np.array(lat)
        lon = np.array(lon)
        sza = np.array(sza)
        ea = np.array(ea)
        pa = np.array(pa)

        image = fits.ImageHDU(name='wavelengths')
        image.data = wavelengths
        columns.append(image)

        image = fits.ImageHDU(name='ls')
        image.data = ls
        columns.append(image)

        image = fits.ImageHDU(name='lat')
        image.data = lat
        columns.append(image)

        image = fits.ImageHDU(name='lon')
        image.data = lon
        columns.append(image)

        image = fits.ImageHDU(name='sza')
        image.data = sza
        columns.append(image)

        image = fits.ImageHDU(name='ea')
        image.data = ea
        columns.append(image)

        image = fits.ImageHDU(name='pa')
        image.data = pa
        columns.append(image)


if __name__ == '__main__':
    p = '/media/kyle/Samsung_T5/IUVS_data'
    iuvsp = IUVSPixels(p)
    iuvsp.make_gale_fits(7204, 7205, 'foo', 'foo')
    #print(a)
    #print(b)
    #print(c)
