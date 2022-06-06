"""Cross-talk rate calculation

Calculate how many zeroes (cross-talk) and valid timestamps were measured.
The module is used for calculation of cross-talk rate based on the data from
multiple data files/acquistion windows. Works with both 'txt' and '.dat'
data files.

Works with both 'txt' and 'dat' files. Output is a '.csv' file with number
of cross-talk zeros and valid timestamps and a separate '.csv' file with data
for a plot.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * cross_talk_rate - calculates the cross-talk rate
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import functions.unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary"

os.chdir(path)

output = []

if "binary" in path:
    # find all data files
    DATA_FILES = glob.glob('*acq*'+'*.dat*')
    # lists for output that will be saved to .csv

    for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
        # unpack data from the txt file into a
        # matrix 256 x data_lines*N_of_cycles
        data_matrix = f_up.unpack_binary_10(DATA_FILES[r])
        for i in range(len(data_matrix)-1):  # 256-1=255 differences
            p = 0  # number of acq cycle
            for j in range(len(data_matrix[0])):  # 10*11999
                if j % 10 == 0:
                    p = p + 1  # next acq cycle
                if data_matrix[i][j] == -1:
                    continue
                for k in range(10):  # 10 lines of data / acq cycle
                    # calculate difference between 'i' and 'i+1' rows
                    # writting in the new matrix data_diff is always
                    # happening in positions 0:9, while subrahend moves
                    # with the acqusition cycle
                    n = 10*(p - 1) + k
                    if data_matrix[i+1][n] == -1:
                        continue
                    else:
                        output.append(np.abs(data_matrix[i][j]
                                             - data_matrix[i+1][n]))
        # find zeros and valid timestamps for cross-talk rate
        output = np.array(output)
        pixel_zeros = output[np.where(output == 0)]
        pixel_valid = output[np.where(output > 0)]

        cross_talk = len(pixel_zeros) / len(pixel_valid) * 100

else:
    # find all data files
    DATA_FILES = glob.glob('*acq*'+'*.txt*')
    # lists for output that will be saved to .csv

    for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
        # unpack data from the txt file into a
        # matrix 256 x data_lines*N_of_cycles
        data_matrix = f_up.unpack_txt_10(DATA_FILES[r])
        for i in range(len(data_matrix)-1):  # 256-1=255 differences
            p = 0  # number of acq cycle
            for j in range(len(data_matrix[0])):  # 10*11999
                if j % 10 == 0:
                    p = p + 1  # next acq cycle
                if data_matrix[i][j] == -1:
                    continue
                for k in range(10):  # 10 lines of data / acq cycle
                    # calculate difference between 'i' and 'i+1' rows
                    # writting in the new matrix data_diff is always
                    # happening in positions 0:9, while subrahend moves
                    # with the acqusition cycle
                    n = 10*(p - 1) + k
                    if data_matrix[i+1][n] == -1:
                        continue
                    else:
                        output.append(np.abs(data_matrix[i][j]
                                             - data_matrix[i+1][n]))
        # find zeros and valid timestamps for cross-talk rate
        output = np.array(output)
        pixel_zeros = output[np.where(output == 0)]
        pixel_valid = output[np.where(output > 0)]

        cross_talk = len(pixel_zeros) / len(pixel_valid) * 100
