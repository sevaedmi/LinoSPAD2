"""Script for calculating and plotting hit rates in each pixel.

"""

import glob
import os

import numpy as np
from matplotlib import pyplot as plt

from functions import unpack as f_up

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"
    "3 ms acq window"
)

os.chdir(path)

filename = glob.glob("*.dat*")[0]

data = f_up.unpack_binary_512(filename)

valid_timestamps = np.zeros((100, len(data)))
rate = np.zeros((100, len(data)))

for i, row in enumerate(data):
    for j in range(100):
        valid_timestamps[i][j] = len(np.where(row > 0)[0])
        rate[i][j] = valid_timestamps[i] / 3 * 1e3


plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel("Pixel nubmer [-]")
plt.ylabel("Rate [Hz]")
plt.title("{}".format(filename))
plt.plot(rate, "o", color="salmon")
