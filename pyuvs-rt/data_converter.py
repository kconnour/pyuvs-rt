"""This module has tools to create more usable files from ascii files."""
import linecache
import numpy as np
from astropy.io import fits


class L1CTxt:
    """A class that can extract all data from an IUVS level 1c txt file.

    It extracts the various properties from an IUVS l1c text file and stores
    them as numpy arrays. It acts somewhat analogously to a .fits file.

    """
    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path
            Absolute path to the IUVS .txt file.

        """
        self.file_path = file_path
        self.n_integrations, self.n_positions, self.n_wavelengths = \
            self._extract_observation_shape_from_file()
        self.pixel_info, self.spectral_info = self._extract_pixel_data()

    def _extract_observation_shape_from_file(self) -> tuple[int, int, int]:
        file_shape_line_num = 4
        file_shape = self._extract_info_from_line(file_shape_line_num, int)
        return file_shape[0], file_shape[1], file_shape[2]

    def _extract_info_from_line(self, line_number: int, dtype: object) \
            -> np.ndarray:
        line = linecache.getline(self.file_path, line_number)
        return np.fromstring(line, dtype=dtype, sep=' ')

    def _extract_pixel_data(self) -> tuple[np.ndarray, np.ndarray]:
        pixel_shape = (self.n_integrations, self.n_positions)
        spectral_shape = pixel_shape + (self.n_wavelengths,)
        pixel_info = np.zeros(pixel_shape, dtype=object)
        spectral_info = np.zeros(spectral_shape, dtype=object)

        for line_number, line in enumerate(open(self.file_path, 'r')):
            if self._is_file_header_line(line_number):
                continue
            line = self._extract_info_from_line(line_number, float)
            if self._is_pixel_info_line(line_number):
                pix_info = _PixelInfo(line)
                pixel_info[pix_info.integration, pix_info.position] = \
                    pix_info
            else:
                spec_info = _SpectralInfo(line)
                wav_index = self._make_wavelength_index(line_number)
                spectral_info[pix_info.integration, pix_info.position,
                              wav_index] = spec_info

        return pixel_info, spectral_info

    @staticmethod
    def _is_file_header_line(line_number: int) -> bool:
        return line_number < 6

    def _is_pixel_info_line(self, line_number: int) -> bool:
        return (line_number - 6) % (self.n_wavelengths + 1) == 0

    def _make_wavelength_index(self, line_number) -> int:
        return (line_number - 6) % (self.n_wavelengths + 1) - 1

    def _make_latitude(self):
        latitude = np.zeros((self.n_integrations, self.n_positions))
        for i in range(self.n_integrations):
            for j in range(self.n_positions):
                latitude[i, j] = self.pixel_info[i, j].latitude


class _PixelInfo:
    def __init__(self, pixel_info: np.ndarray):
        self.integration = int(pixel_info[0] - 1)
        self.position = int(pixel_info[1] - 1)
        self.latitude = pixel_info[2:7]
        self.longitude = pixel_info[7:12]
        self.tangent_altitude = pixel_info[12]
        self.local_time = pixel_info[13]
        self.solar_zenith_angle = pixel_info[14]
        self.emission_angle = pixel_info[15]
        self.phase_angle = pixel_info[16]
        self.solar_longitude = pixel_info[17]


class _SpectralInfo:
    def __init__(self, spectral_info: np.ndarray):
        self.wavelength = spectral_info[0]
        self.solar_flux = spectral_info[1]
        self.reflectance = spectral_info[2]
        self.uncertainty = spectral_info[3]


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
