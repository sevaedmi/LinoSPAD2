"""Calculate how many zeroes (cross-talk) and valid timestamps were measured
in a single acq window. The module is used for calculation of cross-talk rate
based on the data from multiple data files/acquistion windows.
"""

import numpy as np
from tqdm import tqdm
from functions.unpack import unpack_txt_10


def zeros_to_valid(filename):
    # unpack data from the txt file into a matrix 256 x data_lines*N_of_cycles
    Data_matrix = unpack_txt_10.unpack(filename)

    # matrix for timestamp differences
    Data_diff = np.zeros((len(Data_matrix)-1, len(Data_matrix[0])))

    # calculate the timestamp differences
    for i in tqdm(range(len(Data_matrix[0]))):  # take a single column from the data
    # TODO: not a single slice, but whole acq window!!!!!
        j = 0  # row number
        while j < 255:
            # if both numbers in the neighboring rows in the original data are
            # non valid, cut it out from the cross talk calculation
            if Data_matrix[j][i] == -1 and Data_matrix[j+1][i] == -1:
                Data_diff[j][i] = -1
            else:
                Data_diff[j][i] = np.abs(Data_matrix[j][i]
                                         - Data_matrix[j+1][i])
            j = j+1

    # Calculate cross talk in %
    zeros = len(np.where(Data_diff == 0)[0])
    valid_timestamps = len(np.where(Data_matrix >= 0)[0])

    return zeros, valid_timestamps
