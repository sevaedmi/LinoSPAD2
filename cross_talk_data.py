"""Calculate cross-talk from real data.
Output is cross-talk rate in %.
"""

from functions.unpack import unpack_txt
import numpy as np

# absolute path to the data file "acq ... .txt"
file = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "40 ns window, 20 MHz clock, 10 cycles/acq_220512_150911.txt"

# unpack data from the txt file into a matrix 256 x 512*N_of_cycles
Data_matrix = unpack_txt.unpack(file)

# matrix for timestamp differences
Data_diff = np.zeros((len(Data_matrix)-1, len(Data_matrix[0])))

# calculate the timestamp differences
for i in range(len(Data_matrix[0])):  # take a single column from the data
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

number_of_cycles = int(len(Data_matrix[0])/512)
print("Based on data collected during {0} cycles, cross talk rate is "
      "estimated at %.2f %%.".format(number_of_cycles) % Cross_talk_rate)

# =============================================================================
# TODO: 10000 valid timestamps per pixel for good statistics; 1802/256/10=0.7
# valid timestamps per pixel per cycle => 10000*256 - valid timestamps total,
# 10000*256*0.7~=1802000 cycles - intensive for memory! Try r/o 6 rows,
# should suffice; max seen numbers of rows for 20 MHz, 40 ns acq window is 4
