# =============================================================================
# Test script for checking LinoSPAD2 pixel crosstalk
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt

# # Test with random numbers
# Data_matrix = np.sort(np.random.randint(559, size=(256, 512)))
# Diff_matrix = np.zeros((255, 512))

# # calculate time differences between two neighboring rows
# for i in range(len(Data_matrix)-1):
#     Diff_matrix[i] = Data_matrix[i+1] - Data_matrix[i]

# Diff_counts_matrix = np.zeros((255, 21))
# # count the differences
# for i in range(len(Diff_matrix)):
#     for j in range(-10, 11):
#         Diff_counts_matrix[i][j+10] = len(np.where(Diff_matrix[i] == j)[0])

# # =============================================================================

# # Plot to check the results
# plt.figure(figsize=(16, 10))
# plt.xlabel("Time difference [ps]")
# plt.ylabel("Counts [-]")
# for i in range(len(Diff_counts_matrix)):
#     # for j in range (0, 20):
#     #         plt.plot(j-10, Diff_counts_matrix[i][j], 'o')
#     plt.plot(np.arange(-10, 11, 1), Diff_counts_matrix[i])
# plt.show()

# # Calculate crosstalk in %; at [10] lie time differences of 0
# CT = np.zeros(255)
# for i in range(len(CT)):
#     CT[i] = Diff_counts_matrix[i][10]/np.sum(Diff_counts_matrix[i])*100

# =============================================================================
# Test with real data
# =============================================================================

import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import functions.unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary"

os.chdir(path)

if "binary" in path:
    # find all data files
    DATA_FILES = glob.glob('*acq*'+'*.dat*')
    # lists for output that will be saved to .csv
    zeros_to_save = []
    valid_to_save = []
    pixel_cross_talk_to_save = None

    for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
        # unpack data from the txt file into a
        # matrix 256 x data_lines*N_of_cycles
        data_matrix = f_up.unpack_binary_10(DATA_FILES[r])
        # matrix for timestamp differences
        data_diff = np.zeros((len(data_matrix)-1, len(data_matrix[0]), 10))
        for i in range(len(data_matrix)-1):  # 256-1=255 differences
            p = 0  # number of acq cycle
            for j in range(len(data_matrix[0])):  # 10*11999
                if j % 10 == 0:
                    p = p + 1  # next acq cycle
                for k in range(10):  # 10 lines of data / acq cycle
                    # calculate difference between 'i' and 'i+1' rows
                    # writting in the new matrix data_diff is always
                    # happening in positions 0:9, while subrahend moves with
                    # the acqusition cycle
                    n = 10*(p - 1) + k
                    if data_matrix[i][j] == -1 or data_matrix[i+1][n] == -1:
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
            pixel_cross_talk[i] = pixel_zeros[i] / pixel_valid[i] * 100  # in %

        zeros_total = int(np.sum(pixel_zeros))
        valid_total = int(np.sum(pixel_valid))
        zeros_to_save.append(zeros_total)
        valid_to_save.append(valid_total)
        if pixel_cross_talk_to_save is None:
            pixel_cross_talk_to_save = pixel_cross_talk
        else:
            pixel_cross_talk_to_save = np.column_stack(
                (pixel_cross_talk_to_save, pixel_cross_talk))

else:
    # find all data files
    DATA_FILES = glob.glob('*acq*'+'*.txt*')
    # lists for output that will be saved to .csv
    zeros_to_save = []
    valid_to_save = []
    pixel_cross_talk_to_save = None

    for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
        # unpack data from the txt file into a
        # matrix 256 x data_lines*N_of_cycles
        data_matrix = f_up.unpack_txt_10(DATA_FILES[r])
        # matrix for timestamp differences
        data_diff = np.zeros((len(data_matrix)-1, len(data_matrix[0]), 10))
        for i in range(len(data_matrix)-1):  # 256-1=255 differences
            p = 0  # number of acq cycle
            for j in range(len(data_matrix[0])):  # 10*11999
                if j % 10 == 0:
                    p = p + 1  # next acq cycle
                for k in range(10):  # 10 lines of data / acq cycle
                    # calculate difference between 'i' and 'i+1' rows
                    # writting in the new matrix data_diff is always
                    # happening in positions 0:9, while subrahend moves with
                    # the acqusition cycle
                    n = 10*(p - 1) + k
                    if data_matrix[i][j] == -1 or data_matrix[i+1][n] == -1:
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
            pixel_cross_talk[i] = pixel_zeros[i] / pixel_valid[i] * 100  # in %

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

# save the cross-talk rate by pixel for plot

plot_headers = ['Cross-talk rate in %']

data_for_plot = pd.DataFrame(data=pixel_cross_talk_to_save)
data_for_plot.to_csv("Cross-talk by pixel to plot.csv")
