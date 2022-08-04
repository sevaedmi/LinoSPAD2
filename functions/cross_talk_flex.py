"""Script for calculation of the cross-talk rate.

Calculate how many zeroes (cross-talk) and valid timestamps were measured.
The module is used for calculation of cross-talk rate based on the data from
multiple data files/acquistion windows.

Output is a '.csv' file with number of cross-talk zeros and valid timestamps
and a separate '.csv' file with data for a plot. The average cross-talk rate
is returned as int.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * cross_talk_rate - calculates the cross-talk rate

"""

import glob
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from functions import unpack as f_up


def cross_talk_rate(path, lines_of_data: int = 512):
    '''Functions for calculating both the average cross-talk across all pixels
    and the cross-talk of each individual pixel. The average cross-talk is
    returned, while both the average, number of average valid timestamps per
    pixel, and pixel cross-talk rates are saved into a 'csv' file.

    Parameters
    ----------
    path : str
        Path to the data files.
    lines_of_data : int, optional
        Points of data per acq cycle per pixel. The default is 512.

    Returns
    -------
    cross_talk_output : int
        Average cross-talk rate across all pixels.

    '''
    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    zeros_to_save = []
    valid_to_save = []
    pixel_cross_talk_to_save = None

    for r in tqdm(range(len(DATA_FILES)), desc="Calculating: "):

        lod = lines_of_data

        data_matrix = f_up.unpack_binary_flex(DATA_FILES[r], lines_of_data)

        data_diff = np.zeros((len(data_matrix)-1, len(data_matrix[0]),
                             lines_of_data))

        for i in range(len(data_matrix)-1):  # 256-1=255 differences
            acq = 0  # number of acq cycle
            for j in range(len(data_matrix[0])):
                if j % lod == 0:
                    acq = acq + 1  # next acq cycle
                for k in range(lod):  # 'lod' lines of data / acq cycle
                    # calculate difference between 'i' and 'i+1' rows
                    # writting in the new matrix data_diff is always
                    # happening in positions 0:9, while subtrahend moves
                    # with the acqusition cycle
                    n = lod*(acq - 1) + k
                    if data_matrix[i][j] == -1 or \
                       data_matrix[i+1][n] == -1:
                        data_diff[i][j][k] = -1
                    else:
                        data_diff[i][j][k] = np.abs(data_matrix[i][j]
                                                    - data_matrix[i+1][n])
    # find zeros and valid timestamps for cross-talk rate

        pixel_zeros = np.zeros(len(data_diff))
        pixel_valid = np.zeros(len(data_diff))
        pixel_cross_talk = np.zeros(len(data_diff))

        for i in range(len(data_diff)):
            pixel_zeros[i] = len(np.where(data_diff[i] == 0)[0])
            pixel_valid[i] = len(np.where(data_diff[i] > 0)[0])
            pixel_cross_talk[i] = pixel_zeros[i] / pixel_valid[i] * 100

        zeros_total = int(np.sum(pixel_zeros))
        valid_total = int(np.sum(pixel_valid))
        zeros_to_save.append(zeros_total)
        valid_to_save.append(valid_total)
        if pixel_cross_talk_to_save is None:
            pixel_cross_talk_to_save = pixel_cross_talk
        else:
            pixel_cross_talk_to_save = np.column_stack(
                (pixel_cross_talk_to_save, pixel_cross_talk))

    print("\nCalculating the cross-talk rate and saving the data into a"
          "'.csv' file.")

    # number of files with data, each contains data from 11999 acquisition
    # cycles
    number_of_acq_cycles = len(data_matrix[0]) / lod * len(DATA_FILES)

    average_valid_timestamps = np.sum(valid_to_save) / 256

    # cross-talk rate is calculated as zero values divided by total number of
    # valid timestamps (>0)
    cross_talk_output = np.sum(zeros_to_save) / np.sum(valid_to_save) * 100

    # save the number of cross-talk zeros, number of valid timestamps
    # from the original data file, the calculated cross-talk rate, the number
    # of acquisition cycles, and the average number of valid timestamps per
    # pixel in a '.csv' file
    output_to_save = np.zeros((len(DATA_FILES), 5))
    for i in range(len(output_to_save)):
        output_to_save[i][0] = zeros_to_save[i]
        output_to_save[i][1] = valid_to_save[i]
    output_to_save[0][2] = number_of_acq_cycles
    output_to_save[0][3] = average_valid_timestamps
    output_to_save[0][4] = cross_talk_output

    output_headers = ['Number of cross-talk zeros',
                      'Number of valid timestamps',
                      'Number of acq cycles',
                      'Average of valid timestamps per pixel',
                      'Cross-talk rate in %']

    output_to_csv = pd.DataFrame(data=output_to_save, columns=output_headers)

    # save the data into the 'results' folder
    try:
        os.chdir("results")
    except Exception:
        os.mkdir("results")
        os.chdir("results")
    output_to_csv.to_csv("Cross-talk_results.csv")
    print("\nData are saved in the 'Cross-talk_results.csv' that can be found"
          "in the folder 'results'.")

    # save the cross-talk rate by pixel for plot

    data_for_plot = pd.DataFrame(data=pixel_cross_talk_to_save)
    data_for_plot.to_csv("Cross-talk by pixel to plot.csv")
    print("\nData for plot are saved in the 'Cross-talk by pixel to plot.csv'"
          "that can be found in the folder 'results'.")

    return cross_talk_output
