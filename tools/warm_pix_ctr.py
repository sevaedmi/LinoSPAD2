""" Script for calculating cross-talk rate based on warm/hot pixels.

"""

import os
import glob
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from functions import unpack as f_up


path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ar lamp/FW 2208"

lines_of_data = 512

pix1 = (14, 15, 16)
pix2 = (51, 52, 53)
pix3 = (92, 93, 94)
pix4 = (235, 236, 237)

os.chdir(path)

filename = glob.glob("*.dat*")[0]
data = f_up.unpack_binary_flex(filename, 512)

# data_trio = np.vstack((data[pix1[0]], data[pix1[1]], data[pix1[2]]))
# data_trio = np.vstack((data[pix2[0]], data[pix2[1]], data[pix2[2]]))
# data_trio = np.vstack((data[pix3[0]], data[pix3[1]], data[pix3[2]]))
# data_trio = np.vstack((data[pix4[0]], data[pix4[1]], data[pix4[2]]))

data_trio = np.vstack((data[pix1[0]], data[pix1[1]]))

minuend = len(data_trio)
timestamps_total = len(data_trio[0])
subtrahend = len(data_trio)
timestamps = lines_of_data

output = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(timestamps_total):
        if j % lines_of_data == 0:
            acq = acq + 1  # next acq cycle
        if data_trio[i][j] == -1:
            continue
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 53-45
            for p in range(timestamps):
                n = lines_of_data*(acq-1) + p
                if data_trio[k][n] == -1:
                    continue
                elif data_trio[i][j] - data_trio[k][n] > 2.5e3:
                    continue
                elif data_trio[i][j] - data_trio[k][n] < -2.5e3:
                    continue
                else:
                    output.append(data_trio[i][j]
                                  - data_trio[k][n])

if "Ne" and "540" in path:
    chosen_color = "seagreen"
elif "Ne" and "656" in path:
    chosen_color = "orangered"
elif "Ar" in path:
    chosen_color = "mediumslateblue"
else:
    chosen_color = "salmon"

try:
    bins = np.arange(np.min(output), np.max(output),
                     17.857)
except Exception:
    print(1)
plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel('\u0394t [ps]')
plt.ylabel('Timestamps [-]')
n, b, p = plt.hist(output, bins=bins, color=chosen_color)

# find position of the histogram peak
try:
    n_max = np.argmax(n)
    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2,
                     ".2f")
except Exception:
    arg_max = None
    pass

plt.title('Warm pixel {pix}-{p1}\nPeak position: {peak}'.format(pix=14, p1=15,
                                                            peak=arg_max))
# plt.title('Warm pixel {pix}\nPeak position: {peak}'.format(pix=236,
#                                                             peak=arg_max))

try:
    os.chdir("results/warm pixel cross talk")
except Exception:
    os.mkdir("results/warm pixel cross talk")
    os.chdir("results/warm pixel cross talk")

plt.savefig("{pix}-{p1}.png".format(pix=14, p1=15))
# plt.savefig("{pix}.png".format(pix=236))
