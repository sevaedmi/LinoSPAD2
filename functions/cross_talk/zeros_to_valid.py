"""Calculate how many zeroes (cross-talk) and valid timestamps were measured
in a single acq window. The module is used for calculation of cross-talk rate
based on the data from multiple data files/acquistion windows.
"""

import numpy as np
from functions.unpack import unpack_txt_10


def zeros_to_valid(filename):
    # unpack data from the txt file into a matrix 256 x data_lines*N_of_cycles
    Data_matrix = unpack_txt_10.unpack(filename)

    # matrix for timestamp differences
    Data_diff = np.zeros((len(Data_matrix)-1, len(Data_matrix[0]), 10))

    for i in range(len(Data_matrix)-1):  # 256-1=255 differences
        for j in range(len(Data_matrix[0])):  # 10*11999
            for k in range(10):  # 10 lines of data / acq cycle
                if Data_matrix[i][j] == -1 or Data_matrix[i+1][k] == -1:
                    Data_diff[i][j][k] = -1
                else:
                    Data_diff[i][j][k] = np.abs(Data_matrix[i][j]
                                                - Data_matrix[i+1][k])

    # Calculate cross talk in %
    zeros = len(np.where(Data_diff == 0)[0])
    valid_timestamps = len(np.where(Data_matrix >= 0)[0])

    return zeros, valid_timestamps
