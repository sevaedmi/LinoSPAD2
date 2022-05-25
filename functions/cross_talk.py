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
import pandas as pd
import functions.unpack as f_up


def cross_talk_rate(path):
    """Calculates cross-talk rate for LinoSPAD2.

    Parameters
    ----------
    path : str
        Location of the data files

    Returns
    -------
    str
        Cross-talk rate based on number of acquistion cycles analyzed

    """

    os.chdir(path)

    if "binary" in path:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.dat*')
        # lists for output that will be saved to .csv
        zeros_to_save = []
        valid_to_save = []

        for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            # unpack data from the txt file into a
            # matrix 256 x data_lines*N_of_cycles
            data_matrix = f_up.unpack_binary_10(DATA_FILES[r])
            # matrix for timestamp differences
            data_diff = np.zeros((len(data_matrix)-1, len(data_matrix[0]), 10))
            for i in range(len(data_matrix)-1):  # 256-1=255 differences
                for j in range(len(data_matrix[0])):  # 10*11999
                    for k in range(10):  # 10 lines of data / acq cycle
                        # calculate difference between 'i' and 'i+1' rows
                        if data_matrix[i][j] == -1 or data_matrix[i+1][k] == -1:
                            data_diff[i][j][k] = -1
                        else:
                            # TODO: F-UP GRADE ERROR - subtrahends are only
                            # the first 10 points of data. Fix the cycle
                            # to follow the acquistion window with 
                            # 'for k in range(j, j+9)' with smth for 'j' to
                            # keep track of the current acq window.
                            data_diff[i][j][k] = np.abs(data_matrix[i][j]
                                                        - data_matrix[i+1][k])
            # find zeros and valid timestamps for cross-talk rate
            zeros = len(np.where(data_diff == 0)[0])
            valid_timestamps = len(np.where(data_matrix >= 0)[0])
            zeros_to_save.append(zeros)
            valid_to_save.append(valid_timestamps)

    else:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.txt*')
        # lists for output that will be saved to .csv
        zeros_to_save = []
        valid_to_save = []

        for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            # unpack data from the txt file into a
            # matrix 256 x data_lines*N_of_cycles
            data_matrix = f_up.unpack_txt_10(DATA_FILES[r])
            # matrix for timestamp differences
            data_diff = np.zeros((len(data_matrix)-1, len(data_matrix[0]), 10))
            for i in range(len(data_matrix)-1):  # 256-1=255 differences
                for j in range(len(data_matrix[0])):  # 10*11999
                    for k in range(10):  # 10 lines of data / acq cycle
                        # calculate difference between 'i' and 'i+1' rows
                        if data_matrix[i][j] == -1 or data_matrix[i+1][k] == -1:
                            data_diff[i][j][k] = -1
                        else:
                            data_diff[i][j][k] = np.abs(data_matrix[i][j]
                                                        - data_matrix[i+1][k])
            # find zeros and valid timestamps for cross-talk rate
            zeros = len(np.where(data_diff == 0)[0])
            valid_timestamps = len(np.where(data_matrix >= 0)[0])
            zeros_to_save.append(zeros)
            valid_to_save.append(valid_timestamps)

    print("\nCalculating the cross-talk rate and saving the data into a"
          "'.csv' file.")
    # cross-talk rate is calculated as zero values divided by total number of
    # valid timestamps (>0)
    cross_talk_output = np.sum(zeros_to_save) / np.sum(valid_to_save) * 100

    number_of_acq_cycles = 11999*len(DATA_FILES)  # number of files with data,
    # each contains data from 11999 acquisition cycles

    average_valid_timestamps = np.sum(valid_to_save) / 256

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
                      'Cross-talk rate in %%']

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

    # TODO: comment wth is going on; fix the code - dimensions are wrong
    # and FYI np.unique(data_diff) gives bizarre output, probably due to
    # errors above in the filling of the data_diff[2]
    pixels_all = np.where(data_diff == 0)[0]
    pixels = np.unique(pixels_all)
    pixel_zeros = np.zeros(len(data_diff))
    pixel_valid = np.zeros(len(data_diff[0]))
    cross_talk_pixel = np.zeros(len(pixels))

    for i in tqdm(range(len(pixel_zeros)), desc='Collecting data for plot'):
        pixel_zeros[i] = len(np.where(pixels_all == i)[0])
        pixel_valid[i] = len(np.where(data_diff[i] > 0)[0])
        cross_talk_pixel[i] = pixel_zeros[i] / pixel_valid[i]

    data_for_plot = np.zeros(len(pixels), len(pixels))
    for i in range(len(data_for_plot)):
        data_for_plot[i][0] = pixels[i]
        data_for_plot[i][1] = cross_talk_pixel[i]

    plot_headers = ['Pixel', 'Cross-talk rate']

    data_for_plot = pd.DataFrame(data=data_for_plot, columns=plot_headers)
    data_for_plot.to_csv("Cross-talk by pixel to plot.csv")

    # TODO: add plot, pixel vs cross-talk rate to see the cross-talk
    # distribution in the sensor
