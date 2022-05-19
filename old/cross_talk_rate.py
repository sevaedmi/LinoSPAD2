"""Calculate cross-talk from a single file of ~107 MB of data.
Output is cross-talk rate in %.
"""
import numpy as np
from tqdm import tqdm
from functions.unpack import unpack_txt_10

# absolute path to the data file "acq ... .txt"
FILENAME = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "all_data.txt"

# unpack data from the txt file into a matrix 256 x data_lines*N_of_cycles
Data_matrix = unpack_txt_10.unpack(FILENAME)

# matrix for timestamp differences
Data_diff = np.zeros((len(Data_matrix)-1, len(Data_matrix[0])))

# calculate the timestamp differences
for i in tqdm(range(len(Data_matrix[0])), desc='Calculating timestamps'
              ' differences'):  # take a single column from the data
    j = 0  # row number
    while j < 255:
        # if both numbers in the neighboring rows in the original data are non
        # valid, cut it out from the cross talk calculation
        if Data_matrix[j][i] == -1 and Data_matrix[j+1][i] == -1:
            Data_diff[j][i] = -1
        else:
            Data_diff[j][i] = np.abs(Data_matrix[j][i] - Data_matrix[j+1][i])
        j = j+1

# Calculate cross talk in %
Cross_talk_zeros = len(np.where(Data_diff == 0)[0])
Valid_timestamps = len(np.where(Data_matrix >= 0)[0])

Cross_talk_rate = Cross_talk_zeros / Valid_timestamps * 100

LINES_OF_DATA = 10  # change to the appropriate number of lines in the txt

number_of_cycles = int(len(Data_matrix[0])/LINES_OF_DATA)
print("Based on data collected during {0} cycles, cross talk rate is "
      "estimated at %.5f %%.".format(number_of_cycles) % Cross_talk_rate)
