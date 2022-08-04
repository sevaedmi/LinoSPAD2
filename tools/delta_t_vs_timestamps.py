"""Script for plotting delta_t vs timestamps of each of the pixels in the pair.

"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ne lamp ext trig/setup 2/3.99 ms acq window"

os.chdir(path)

filename = glob.glob("*.dat*")[0]

data = f_up.unpack_binary_512(filename)

data_cut_1_pix = data[252]
data_cut_2_pix = data[253]

data_cut = np.vstack((data_cut_1_pix, data_cut_2_pix))

minuend = len(data_cut)-1  # i=255
lines_of_data = len(data_cut[0])  # j=10*11999 (lines of data
# * number of acq cycles)
subtrahend = len(data_cut)  # k=254
timestamps = 512  # lines of data in the acq cycle

output = []
data_1 = []
data_2 = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(lines_of_data):
        if data_cut[i][j] == -1:
            continue
        if j % 512 == 0:
            acq = acq + 1  # next acq cycle
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 153-45 etc.
            for p in range(timestamps):
                n = 512*(acq-1) + p
                if data_cut[k][n] == -1:
                    continue
                elif data_cut[i][j] - data_cut[k][n] > 4e3:  #
                    continue
                elif data_cut[i][j] - data_cut[k][n] < 2e3:
                    continue
                else:
                    output.append(data_cut[i][j]
                                  - data_cut[k][n])
                    data_1.append(data_cut[i][j])  # save the used timestamps
                    data_2.append(data_cut[k][n])  # for 2d histogram

plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 22})
plt.xlabel("Timestamps [ps]")
plt.ylabel("\u0394t [ps]")
plt.hist2d(data_1, output, bins=(200, 200))
plt.colorbar()
try:
    os.chdir("results/test")
except Exception:
    pass
plt.savefig("delta_t_vs_timestamp_pixel 252.png")

plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 22})
plt.xlabel("Timestamps [ps]")
plt.ylabel("Delta t [ps]")
plt.hist2d(data_2, output, bins=(200, 200))
plt.colorbar()
try:
    os.chdir("results/test")
except Exception:
    pass
plt.savefig("delta_t_vs_timestamp_pixel 253.png")
