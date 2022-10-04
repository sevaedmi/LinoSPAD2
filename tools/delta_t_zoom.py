"""Script for plotting a single zoomed-in plot of delta t between a pair of
pixels.


"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/Ar lamp/FW 2208"

os.chdir(path)

filename = glob.glob('*.dat*')[0]

data = f_up.unpack_binary_512(filename)

# =============================================================================

print("What pair of pixels in the range 251-255 should be analyzed?")
p1 = int(input("Pixel number 1: "))
p2 = int(input("Pixel number 2: "))

data_p1 = data[p1]  # 251st pixel
data_p2 = data[p2]  # 253st pixel

data_pair = np.vstack((data_p1, data_p2))

minuend = len(data_pair)-1  # i=255
lines_of_data = len(data_pair[0])
subtrahend = len(data_pair)  # k=256
timestamps = 512  # lines of data in the acq cycle

output = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(lines_of_data):
        if data_pair[i][j] == -1:
            continue
        if j % 512 == 0:
            acq = acq + 1  # next acq cycle
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 153-45 etc.
            for p in range(timestamps):
                n = 512*(acq-1) + p
                if data_pair[k][n] == -1:
                    continue
                elif data_pair[i][j] - data_pair[k][n] > -3e3:
                    continue
                elif data_pair[i][j] - data_pair[k][n] < -5e3:
                    continue
                else:
                    output.append(data_pair[i][j]
                                  - data_pair[k][n])

bins = np.arange(np.min(output), np.max(output), 17.857)

plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel('delta_t [ps]')
plt.ylabel('Timestamps [-]')
plt.title('Pixels {p1}-{p2}'.format(p1=p1, p2=p2))
plt.hist(output, bins=bins)

try:
    os.chdir("results/delta_t")
except Exception:
    os.mkdir("results/delta_t")
    os.chdir("results/delta_t")
plt.savefig("{name}, pixels {p1}-{p2}, zoom.png".format(name=filename, p1=p1,
                                                        p2=p2))
plt.pause(0.1)
plt.close()
os.chdir("../..")
