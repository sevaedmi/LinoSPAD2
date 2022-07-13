import os
import numpy as np
from matplotlib import pyplot as plt
import glob
from functions import unpack as f_up
from tqdm import tqdm

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "Ne lamp ext trig/setup 2/3 ms acq window"
os.chdir(path)

filename = glob.glob('*.dat*')[0]

data = f_up.unpack_binary_512(filename)

# data_cut = data[-10:]

data_cut_1_pix = data[-5]  # 251st pixel
data_cut_2_pix = data[-3]  # 253st pixel
data_cut_3_pix = data[-2]  # 254st pixel

data_cut = np.vstack((data_cut_1_pix, data_cut_3_pix))

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

# output_sort = np.sort(output)

# output_left = np.where(output_sort > -1e6)[0][0]
# output_right = np.where(output_sort > 1e6)[0][0]

# output_cut = output[output_left:output_right]

# bins = np.linspace(np.min(output_cut), np.max(output_cut), 10)
plt.ion()

bins = np.arange(np.min(output), np.max(output), 17.857*10)
plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 22})
plt.xlabel('delta_t [ps]')
plt.ylabel('Timestamps [-]')
plt.title('Pixels 251-254')
plt.hist(output, bins=bins)
