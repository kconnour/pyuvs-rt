import numpy as np


class L1CAscii:
    """Create a .npy file from Franck's l1c ascii files.

    """
    def __init__(self, file):
        self.file = file
        self.n_integrations, self.n_positions, self.n_wavelengths = \
            self.__extract_array_size()
        self.reflectance = self.__make_holder_array()

    def __extract_array_size(self):
        for counter, line in enumerate(self.__open_file()):
            if counter < 3:
                continue
            elif counter == 3:
                split_line = self.__extract_numbers_from_line(line)
                n_integrations = int(split_line[0])
                n_positions = int(split_line[1])
                n_wavelengths = int(split_line[2])
                return n_integrations, n_positions, n_wavelengths

    def __open_file(self):
        return open(self.file, 'r')

    @staticmethod
    def __extract_numbers_from_line(line):
        splits = line.rstrip().split(' ')
        while '' in splits:
            splits.remove('')
        return splits

    def __make_holder_array(self):
        return np.zeros((self.n_integrations, self.n_positions,
                         self.n_wavelengths))

    def create_fits(self):
        self.__fill_array()
        save_filename = self.file.replace('txt', 'npy')
        np.save(save_filename, self.reflectance)

    def __fill_array(self):
        for counter, line in enumerate(self.__open_file()):
            if counter < 5:
                continue
            if (counter - 5) % (self.n_wavelengths + 1) == 0:
                integration_ind, position_ind = \
                    self.__get_indices_from_info_line(line)
            else:
                reflectance = self.__get_reflectance_from_line(line)
                self.reflectance[
                    integration_ind, position_ind, (counter - 5) % (
                                self.n_wavelengths + 1) - 1] = reflectance

    def __get_indices_from_info_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        return int(splits[0]) - 1, int(splits[1]) - 1

    def __get_reflectance_from_line(self, line):
        splits = self.__extract_numbers_from_line(line)
        return float(splits[2])
