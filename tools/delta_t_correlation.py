"""Script for plotting delta t of a pair of pixels vs delta t of another pair.

"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ar lamp/FW 2208"

os.chdir(path)

filename = glob.glob("*.dat*")[0]

all_data = f_up.unpack_binary_flex(filename, 512)

lines_of_data = 512

data_pair1 = np.vstack((all_data[14], all_data[15]))
data_pair2 = np.vstack((all_data[15], all_data[16]))

minuend = len(data_pair1)
timestamps_total = len(data_pair1[0])
subtrahend = len(data_pair1)
timestamps = lines_of_data

output1 = []
output2 = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(timestamps_total):
        if j % lines_of_data == 0:
            acq = acq + 1  # next acq cycle
        if data_pair1[i][j] == -1:
            continue
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 53-45
            for p in range(timestamps):
                n = lines_of_data*(acq-1) + p
                if data_pair1[k][n] == -1:
                    continue
                elif data_pair1[i][j] - data_pair1[k][n] > 2.5e3:
                    continue
                elif data_pair1[i][j] - data_pair1[k][n] < -2.5e3:
                    continue
                else:
                    output1.append(data_pair1[i][j]
                                  - data_pair1[k][n])

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(timestamps_total):
        if j % lines_of_data == 0:
            acq = acq + 1  # next acq cycle
        if data_pair2[i][j] == -1:
            continue
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 53-45
            for p in range(timestamps):
                n = lines_of_data*(acq-1) + p
                if data_pair2[k][n] == -1:
                    continue
                elif data_pair2[i][j] - data_pair2[k][n] > 2.5e3:
                    continue
                elif data_pair2[i][j] - data_pair2[k][n] < -2.5e3:
                    continue
                else:
                    output2.append(data_pair2[i][j]
                                  - data_pair2[k][n])

output2 = output2[395:]

# histogram of first pair
plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel('\u0394t [ps]')
plt.ylabel('Timestamps [-]')
plt.title("Pair 14-15")
plt.hist(output1, bins=100)

# histogram of second pair
plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel('\u0394t [ps]')
plt.ylabel('Timestamps [-]')
plt.title("Pair 15-16")
plt.hist(output2, bins=100)

# 2d histogram of one pair vs the other
plt.figure(figsize=(16, 16))
plt.rcParams.update({"font.size": 22})
plt.xlabel('\u0394t, first pair [ps]')
plt.ylabel('\u0394t, second pair [ps]')
plt.hist2d(output1, output2, bins=(100, 100))
plt.title("Pair 14-15 vs 15-16")
plt.show()
