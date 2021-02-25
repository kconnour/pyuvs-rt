"""gale_crater contains classes to pull relevant info from multiple spacecraft
about Gale Crater
"""

import os
import numpy as np
import pandas as pd


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
