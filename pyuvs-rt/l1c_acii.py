import numpy as np
from astropy.io import fits


class L1CAscii:
    """Create a .npy file from Franck's l1c ascii files.

    """
    def __init__(self, file: str) -> None:
        self.file = file
        self.n_integrations, self.n_positions, self.n_wavelengths = \
            self.__extract_observation_size()

    def __extract_observation_size(self):
        for counter, line in enumerate(self.__open_file()):
            if counter < 3:
                continue
            elif counter == 3:
                split_line = self.__extract_numbers_from_line(line)
                n_integrations = int(split_line[0])
                n_positions = int(split_line[1])
                n_wavelengths = int(split_line[2])
                return n_integrations, n_positions, n_wavelengths

    def make_pixel_latitude(self):
        pixel_latitudes = np.zeros((self.n_integrations, self.n_positions, 5))
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_pixel_indices_from_info_line(line)
                latitudes = self.__get_latitudes_from_info_line(line)
                pixel_latitudes[integration_ind, position_ind, :] = latitudes
        return pixel_latitudes

    def make_pixel_longitude(self):
        pixel_longitudes = np.zeros((self.n_integrations, self.n_positions, 5))
        for counter, line in enumerate(self.__open_file()):
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

    def __open_file(self):
        return open(self.file, 'r')

    @staticmethod
    def __extract_numbers_from_line(line):
        splits = line.rstrip().split(' ')
        while '' in splits:
            splits.remove('')
        return splits

    def __get_pixel_indices_from_info_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        # -1 since Franck uses 1-based indexing, not 0
        return int(splits[0]) - 1, int(splits[1]) - 1

    def __get_latitudes_from_info_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        return splits[2:7]

    def __get_longitudes_from_info_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        return splits[7:12]

    def __get_tangent_altitude_from_info_line(self, line) -> float:
        splits = self.__extract_numbers_from_line(line)
        return splits[12]

    def __get_local_time_from_info_line(self, line) -> float:
        splits = self.__extract_numbers_from_line(line)
        return splits[13]

    def __get_solar_zenith_angle_from_info_line(self, line) -> float:
        splits = self.__extract_numbers_from_line(line)
        return splits[14]

    def __get_emission_angle_from_info_line(self, line) -> float:
        splits = self.__extract_numbers_from_line(line)
        return splits[15]

    def __get_phase_angle_from_info_line(self, line) -> float:
        splits = self.__extract_numbers_from_line(line)
        return splits[16]

    def __get_reflectance_from_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        return float(splits[2])


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
