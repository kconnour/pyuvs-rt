"""This module has tools to create more usable files from ascii files."""
import linecache
import numpy as np
from astropy.io import fits


class L1CAscii:
    """A class that can extract all data from the level 1c txt files.

    It contains methods to extract the various properties from an l1c file and
    store them as numpy arrays.

    """
    def __init__(self, file: str):
        self.file = file
        self.n_integrations, self.n_positions, self.n_wavelengths = \
            self._extract_observation_shape_from_file()

    def _extract_observation_shape_from_file(self) -> tuple[int, int, int]:
        file_shape_line_num = 3
        file_shape = self._extract_info_from_line(file_shape_line_num, int)
        return file_shape[0], file_shape[1], file_shape[2]

    def _open_file(self):
        return open(self.file, 'r')

    def _extract_info_from_line(self, line_number: int, dtype: object) -> np.ndarray:
        line = linecache.getline(self.file, line_number)
        return np.fromstring(line, dtype=dtype, sep=' ')

    def make_pixel_latitude(self):
        array_shape = (self.n_integrations, self.n_positions, 5)
        pixel_latitude = self._make_empty_array(array_shape)

        for counter, line in enumerate(self._open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                line = self._extract_info_from_line(counter, float)
                pixel_info = _PixelInfo(line)
                pixel_latitude[pixel_info.integration, pixel_info.position, :] \
                    = pixel_info.latitude
        return pixel_latitude

    @staticmethod
    def _make_empty_array(shape: tuple) -> np.ndarray:
        return np.zeros(shape)








    def make_pixel_longitude(self):
        pixel_longitudes = np.zeros((self.n_integrations, self.n_positions, 5))
        for counter, line in enumerate(self._open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                longitudes = self.__get_longitudes_from_info_line(line)
                pixel_longitudes[integration_ind, position_ind, :] = longitudes
        return pixel_longitudes

    def make_tangent_altitude(self):
        pixel_tan_altitude = np.zeros((self.n_integrations, self.n_positions))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                tan_alt = self.__get_tangent_altitude_from_info_line(line)
                pixel_tan_altitude[integration_ind, position_ind] = tan_alt
        return pixel_tan_altitude

    def make_local_time(self):
        pixel_local_time = np.zeros((self.n_integrations, self.n_positions))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                local_time = self.__get_local_time_from_info_line(line)
                pixel_local_time[integration_ind, position_ind] = local_time
        return pixel_local_time

    def make_solar_zenith_angle(self):
        sza = np.zeros((self.n_integrations, self.n_positions))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                szas = self.__get_solar_zenith_angle_from_info_line(line)
                sza[integration_ind, position_ind] = szas
        return sza

    def make_emission_angle(self):
        emission_angle = np.zeros((self.n_integrations, self.n_positions))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                eas = self.__get_emission_angle_from_info_line(line)
                emission_angle[integration_ind, position_ind] = eas
        return emission_angle

    def make_phase_angle(self):
        phase_angle = np.zeros((self.n_integrations, self.n_positions))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                pas = self.__get_phase_angle_from_info_line(line)
                phase_angle[integration_ind, position_ind] = pas
        return phase_angle

    def make_reflectance(self):
        reflectance = np.zeros((self.n_integrations, self.n_positions,
                                self.n_wavelengths))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
            else:
                rfl = self.__get_reflectance_from_line(line)
                wavelength_index = (counter - 5) % (self.n_wavelengths + 1) - 1
                reflectance[integration_ind, position_ind, wavelength_index] = rfl
        return reflectance

    def __get_reflectance_from_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        return float(splits[2])


class _PixelInfo:
    def __init__(self, pixel_info: np.ndarray):
        self.integration = int(pixel_info[1] - 1)
        self.position = int(pixel_info[1] - 1)
        self.latitude = pixel_info[2:7]
        self.longitude = pixel_info[7:12]
        self.tangent_altitude = pixel_info[12]
        self.local_time = pixel_info[13]
        self.solar_zenith_angle = pixel_info[14]
        self.emission_angle = pixel_info[15]
        self.phase_angle = pixel_info[16]
        self.solar_longitude = pixel_info[17]


if __name__ == '__main__':
    pass


class CreateFits:
    """A CreateFits object allows users to make .fits files"""
    def __init__(self, primary_hdu):
        """
        Parameters
        ----------
        primary_hdu: np.ndarray
            The data to go into the primary structure
        """
        self.primary_hdu = primary_hdu
        self.columns = []
        self.__add_primary_hdu()

    def add_image_hdu(self, data, name):
        """Add an ImageHDU to this object

        Parameters
        ----------
        data: np.ndarray
            The data to add to this structure
        name: str
            The name of this ImageHDU

        Returns
        -------
            None
        """
        self.__check_input_is_str(name, 'name')
        self.__check_addition_is_numpy_array(data, name)
        image = fits.ImageHDU(name=name)
        image.data = data
        self.columns.append(image)

    def save_fits(self, save_location, overwrite=True):
        """Save this object as a .fits file

        Parameters
        ----------
        save_location: str
            The location where to save this .fits file
        overwrite: bool
            Denote if this object should overwrite a file with the same name as save_location. Default is True

        Returns
        -------
            None
        """
        self.__check_input_is_str(save_location, 'save_location')
        combined_fits = fits.HDUList(self.columns)
        combined_fits.writeto(save_location, overwrite=overwrite)

    def __add_primary_hdu(self):
        self.__check_addition_is_numpy_array(self.primary_hdu, 'primary')
        hdu = fits.PrimaryHDU()
        hdu.data = self.primary_hdu
        self.columns.append(hdu)

    @staticmethod
    def __check_addition_is_numpy_array(array, name):
        pass

    @staticmethod
    def __check_input_is_str(test_name, input_name):
        if not isinstance(test_name, str):
            raise TypeError(f'{input_name} must be a string.')
