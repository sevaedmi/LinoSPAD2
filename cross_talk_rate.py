""" Calculate cross-talk rate for multiple files of data, each with ~ 107 MB
of data.
"""

import numpy as np
import os
import glob
from tqdm import tqdm
from functions.cross_talk.zeros_to_valid import zeros_to_valid

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "40 ns window, 20 MHz clock, 10 cycles/10 lines of data/txt"
os.chdir(path)

# find all data files
DATA_FILES = glob.glob('*acq*'+'*.txt*')

# for 'zeros_to_valid' output
zeros = []
valid_timestamps = []

for i in tqdm(range(len(DATA_FILES)), desc='Calculating'):
    zero, valid = zeros_to_valid(DATA_FILES[i])
    zeros.append(zero)
    valid_timestamps.append(valid)

# cross-talk rate is calculated as zero values divided by total number of
# valid timestamps (>0)
cross_talk_output = np.sum(zeros) / np.sum(valid_timestamps)

number_of_acq_cycles = 11999*len(DATA_FILES)  # number of files with data,
# each contains data from 11999 acquisition cycles

average_valid_timestamps = np.sum(valid_timestamps) / 256

print("Cross-talk rate is estimated at %.5f %% based on the data collected "
      "from {0} acquisiton cycles with average number of valid timestamps "
      "per pixel of {1}".format(number_of_acq_cycles,
                                average_valid_timestamps) % cross_talk_output)
