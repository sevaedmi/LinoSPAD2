import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "Ar lamp/FW 2208"
)

os.chdir(path)

filename = glob.glob("*.dat*")[0]  # w/o f,p
filename2 = glob.glob("*.dat*")[1]  # w/ p

pix = np.array((157, 158))

data = f_up.unpack_binary_512(filename)
data2 = f_up.unpack_binary_512(filename2)

data_1 = data[pix[0]]
data_2 = data[pix[1]]

data2_1 = data2[pix[0]]
data2_2 = data2[pix[1]]

data_pair1 = np.vstack((data_1, data_2))
data_pair2 = np.vstack((data2_1, data2_2))

# =============================================================================
# FIGURES
# =============================================================================

fig = plt.figure(figsize=(12, 8))
plt.rcParams.update({"font.size": 22})
plt.xlabel("\u0394t [ps]")
plt.ylabel("Timestamps [-]")
plt.title("Pixels {p1}-{p2}, delta t".format(p1=pix[0], p2=pix[1]))

minuend = len(data_pair1) - 1  # i=255
lines_of_data = len(data_pair1[0])
subtrahend = len(data_pair1)  # k=254
timestamps = 512  # lines of data in the acq cycle

output1 = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(lines_of_data):
        if data_pair1[i][j] == -1:
            continue
        if j % 512 == 0:
            acq = acq + 1  # next acq cycle
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 153-45 etc.
            for p in range(timestamps):
                n = 512 * (acq - 1) + p
                if data_pair1[k][n] == -1:
                    continue
                elif data_pair1[i][j] - data_pair1[k][n] > 3e5:  #
                    continue
                elif data_pair1[i][j] - data_pair1[k][n] < -3e5:
                    continue
                else:
                    output1.append(data_pair1[i][j] - data_pair1[k][n])
try:
    bins1 = np.arange(np.min(output1), np.max(output1), 17.857 * 100)
except Exception:
    print("Stopped 1")

plt.figure("fig2")
n, b, p = plt.hist(output1, bins=bins1)

n1_max = np.max(n)

plt.figure(fig)
plt.plot(b[:-1], n / n1_max, color="slateblue", label="Ar, no optics")


output2 = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(lines_of_data):
        if data_pair2[i][j] == -1:
            continue
        if j % 512 == 0:
            acq = acq + 1  # next acq cycle
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 153-45 etc.
            for p in range(timestamps):
                n = 512 * (acq - 1) + p
                if data_pair2[k][n] == -1:
                    continue
                elif data_pair2[i][j] - data_pair2[k][n] > 3e5:  #
                    continue
                elif data_pair2[i][j] - data_pair2[k][n] < -3e5:
                    continue
                else:
                    output2.append(data_pair2[i][j] - data_pair2[k][n])
try:
    bins2 = np.arange(np.min(output2), np.max(output2), 17.857 * 100)
except Exception:
    print("Stopped 2")

plt.figure("fig2")
n, b, p = plt.hist(output2, bins=bins2)

n2_max = np.max(n)

plt.figure(fig)
plt.plot(b[:-1], n / n2_max, color="orange", label="Ar, polarizer")

plt.figure(fig)
plt.legend()
plt.show()
