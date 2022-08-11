"""Script for calculation of the cross-talk rate for pixels with one in
between.

Calculate how many zeroes (cross-talk) and valid timestamps were measured.
The module is used for calculation of cross-talk rate based on the data from
multiple data files/acquistion windows.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * cross_talk_rate - calculates the cross-talk rate for pixels with one
    in between (i, i+2)

"""

import glob
import os
from tqdm import tqdm
from functions import unpack as f_up


def cross_talk_rate(path, lines_of_data: int = 512):
    '''Calculates cross-talk rate for pixels with one in between (i, i+2).

    Parameters
    ----------
    path : str
        Path to the data files.
    lines_of_data : int, optional
        Points of data per acqusition cycle per pixel. The default is 512.

    Returns
    -------
    cross_talk_output : int
        Average cross-talk rate across all pixels.

    '''
    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    zeros_total = 0
    valid_total = 0

    for r in tqdm(range(len(DATA_FILES)), desc="Calculating the average"
                  "cross-talk rate for i and i+2 pixels: "):

        lod = lines_of_data

        data_matrix = f_up.unpack_binary_flex(DATA_FILES[r], lines_of_data)

        for i in range(len(data_matrix)-2):  # 256-1=255 differences
            acq = 0  # number of acq cycle
            for j in range(len(data_matrix[0])):
                if j % lod == 0:
                    acq = acq + 1  # next acq cycle
                if data_matrix[i][j] == -1:
                    continue
                for k in range(lod):  # 'lod' lines of data / acq cycle
                    # calculate difference between 'i' and 'i+1' rows
                    # writting in the new matrix data_diff is always
                    # happening in positions 0:9, while subtrahend moves
                    # with the acqusition cycle
                    n = lod*(acq - 1) + k
                    if data_matrix[i+2][n] == -1:
                        continue
                    elif data_matrix[i][j] - data_matrix[i+2][n] == 0:
                        zeros_total = zeros_total + 1
                    else:
                        valid_total = valid_total + 1

    # cross-talk rate is calculated as zero values divided by total number of
    # differences between valid timestamps (>0)
    cross_talk_output = zeros_total / valid_total * 100

    print("The average cross-talk rate for pixels with one in between is"
          "estimated at: {} %".format(cross_talk_output))

    return cross_talk_output
