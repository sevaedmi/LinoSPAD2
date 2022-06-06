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
    output: numpy 1D array
        Array of all timestamp differences.

    """

    os.chdir(path)

    output = []

    if "binary" in path:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.dat*')
        for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            data_matrix = f_up.unpack_binary_10(DATA_FILES[i])
            # dimensions for matrix of timestamp differences
            minuend = len(data_matrix) - 1  # i=255
            lines_of_data = len(data_matrix[0])  # j=10*11999 (lines of data
            # * number of acq cycles)
            subtrahend = len(data_matrix) - 2  # k=254
            timestamps = 10  # lines of data in the acq cycle

            for i in tqdm(range(minuend)):
                r = 0  # number of acq cycle
                for j in range(lines_of_data):
                    if data_matrix[i][j] == -1:
                        continue
                    if j % 10 == 0:
                        r = r + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 153-45 etc.
                        for p in range(timestamps):
                            n = 10*(r-1) + p
                            if data_matrix[k][n] == -1:
                                continue
                            else:
                                output.append(data_matrix[i][j]
                                              - data_matrix[k][n])
    else:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.txt*')
        for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            data_matrix = f_up.unpack_txt_10(DATA_FILES[i])
            # dimensions for matrix of timestamp differences
            minuend = len(data_matrix) - 1  # i=255
            lines_of_data = len(data_matrix[0])  # j=10*11999 (lines of data
            # * number of acq cycles)
            subtrahend = len(data_matrix) - 2  # k=254
            timestamps = 10

            for i in range(minuend):
                r = 0  # number of acq cycle
                for j in range(lines_of_data):
                    if j % 10 == 0:
                        r = r + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 153-45 etc.
                        for p in range(timestamps):
                            n = 10*(r-1) + p
                            if data_matrix[i][j] == -1 or \
                               data_matrix[k][n] == -1:
                                continue
                            else:
                                output.append(data_matrix[i][j]
                                              - data_matrix[k][n])
    output = np.array(output)
    return output
# TODO: plot(x,y : timestamps difference, counts? coincidence? g^2(0)?)
