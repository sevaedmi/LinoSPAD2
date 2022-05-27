"""Calculate differences in timestamps between all pixels in each acquistion
cycle.

Works with both 'txt' and 'dat' files.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * timestamp_diff - calculates the differences in timestamps between all
    pixels
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import functions.unpack as f_up


def timestamp_diff(path):
    """Calculates the differences in timestamps between all pixels in each
    acquistion cycle.

    Parameters
    ----------
    path : str
        Location of data files from LinoSPAD2.

    Returns
    -------
    TODO: fill in
    TYPE
        DESCRIPTION.

    """

    os.chdir(path)

    if "binary" in path:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.dat*')
        for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            Data_matrix = f_up.unpack_binary_10.unpack_binary(DATA_FILES[i])
            # dimensions for matrix of timestamp differences
            minuend = len(Data_matrix) - 1  # i=255
            lines_of_data = len(Data_matrix[0])  # j=10*11999 (lines of data
            # X number of acq cycles)
            subtrahend = len(Data_matrix) - 2  # k=254
            timestamps = 10

            Data_diff = np.zeros(minuend, lines_of_data, subtrahend,
                                 timestamps)

            for i in range(minuend):
                for j in range(lines_of_data):
                    for k in range(subtrahend):
                        for p in range(timestamps):
                            if Data_matrix[i][j] == -1 or \
                               Data_matrix[k][p] == -1:
                                Data_diff[i][j][k][p] = -1
                            else:
                                Data_diff[i][j][k][p] = Data_matrix[i][j]
                                - Data_matrix[k][p]
    else:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.txt*')
        for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            Data_matrix = f_up.unpack_txt_10.unpack_txt(DATA_FILES[i])
            # dimensions for matrix of timestamp differences
            minuend = len(Data_matrix) - 1  # i=255
            lines_of_data = len(Data_matrix[0])  # j=10*11999 (lines of data
            # X number of acq cycles)
            subtrahend = len(Data_matrix) - 2  # k=254
            timestamps = 10

            Data_diff = np.zeros(minuend, lines_of_data, subtrahend,
                                 timestamps)

            for i in range(minuend):
                for j in range(lines_of_data):
                    for k in range(subtrahend):
                        for p in range(timestamps):
                            if Data_matrix[i][j] == -1 or \
                               Data_matrix[k][p] == -1:
                                Data_diff[i][j][k][p] = -1
                            else:
                                Data_diff[i][j][k][p] = Data_matrix[i][j]
                                - Data_matrix[k][p]

    return  # TODO: resolve the output
# TODO: plot(x,y : timestamps difference, counts? coincidence? g^2(0)?)
