"""Calculate how many zeroes (cross-talk) and valid timestamps were measured
in a single acq window. The module is used for calculation of cross-talk rate
based on the data from multiple data files/acquistion windows.
Works with both 'txt' and '.dat' data files.

The flow of the script:
1) Check what format the data files are in: 'txt' or binary-coded 'bin'
2) Find all data files
3) In a loop, unpack the data file,
4) Calculate timestamp differences between neighboring rows and all timestamps
in a single acquisition window,
5) Save zero values of cross-talk and valid timestamps from the original data
file into a list
6) Calculate cross-talk rate, save the output into a .csv file
"""

import numpy as np
import os
import glob
from tqdm import tqdm
import pandas as pd
import functions.unpack as f_up


def cross_talk_rate(path):
    os.chdir(path)

    if "binary" in path:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.dat*')
        # lists for output that will be saved to .csv
        zeros_to_save = []
        valid_to_save = []

        for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            # unpack data from the txt file into a
            # matrix 256 x data_lines*N_of_cycles
            Data_matrix = f_up.unpack_binary_10.unpack_binary(DATA_FILES[i])
            # matrix for timestamp differences
            Data_diff = np.zeros((len(Data_matrix)-1, len(Data_matrix[0]), 10))
            for i in range(len(Data_matrix)-1):  # 256-1=255 differences
                for j in range(len(Data_matrix[0])):  # 10*11999
                    for k in range(10):  # 10 lines of data / acq cycle
                        # calculate difference between 'i' and 'i+1' rows
                        if Data_matrix[i][j] == -1 or Data_matrix[i+1][k] == -1:
                            Data_diff[i][j][k] = -1
                        else:
                            Data_diff[i][j][k] = np.abs(Data_matrix[i][j]
                                                        - Data_matrix[i+1][k])
            # find zeros and valid timestamps for cross-talk rate
            zeros = len(np.where(Data_diff == 0)[0])
            valid_timestamps = len(np.where(Data_matrix >= 0)[0])
            zeros_to_save.append(zeros)
            valid_to_save.append(valid_timestamps)

    else:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.txt*')
        # lists for output that will be saved to .csv
        zeros_to_save = []
        valid_to_save = []

        for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            # unpack data from the txt file into a
            # matrix 256 x data_lines*N_of_cycles
            Data_matrix = f_up.unpack_txt_10.unpack_txt(DATA_FILES[i])
            # matrix for timestamp differences
            Data_diff = np.zeros((len(Data_matrix)-1, len(Data_matrix[0]), 10))
            for i in range(len(Data_matrix)-1):  # 256-1=255 differences
                for j in range(len(Data_matrix[0])):  # 10*11999
                    for k in range(10):  # 10 lines of data / acq cycle
                        # calculate difference between 'i' and 'i+1' rows
                        if Data_matrix[i][j] == -1 or Data_matrix[i+1][k] == -1:
                            Data_diff[i][j][k] = -1
                        else:
                            Data_diff[i][j][k] = np.abs(Data_matrix[i][j]
                                                        - Data_matrix[i+1][k])
            # find zeros and valid timestamps for cross-talk rate
            zeros = len(np.where(Data_diff == 0)[0])
            valid_timestamps = len(np.where(Data_matrix >= 0)[0])
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
