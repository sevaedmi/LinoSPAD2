"""Script for plotting delta t histogram for specific pairs of pixels.

"""

import os
import numpy as np
from matplotlib import pyplot as plt
import glob
from functions import unpack as f_up
from tqdm import tqdm

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ne lamp ext trig/setup 2/3.99 ms acq window/656 nm"

os.chdir(path)

filename = glob.glob('*.dat*')[0]

data = f_up.unpack_binary_512(filename)

data_251 = data[251]  # 251st pixel
data_252 = data[252]  # 253st pixel
data_253 = data[253]  # 254st pixel
data_254 = data[254]  # 254st pixel
data_255 = data[255]  # 254st pixel

data_cut = np.vstack((data_251, data_252))

minuend = len(data_cut)-1  # i=255
lines_of_data = len(data_cut[0])  # j=10*11999 (lines of data
# * number of acq cycles)
subtrahend = len(data_cut)  # k=254
timestamps = 512  # lines of data in the acq cycle

output = []

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
                elif data_cut[i][j] - data_cut[k][n] > 1e4:  #
                    continue
                elif data_cut[i][j] - data_cut[k][n] < -1e4:
                    continue
                else:
                    output.append(data_cut[i][j]
                                  - data_cut[k][n])

plt.ion()

bins = np.arange(np.min(output), np.max(output), 17.857*10)
plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 22})
plt.xlabel('delta_t [ps]')
plt.ylabel('Timestamps [-]')
plt.title('Pixels 251-254')
plt.hist(output, bins=bins)
