"""Script for plotting histograms of timestamps for single pixels.

"""

import glob
import os
import numpy as np
from functions import unpack as f_up
from matplotlib import pyplot as plt

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
    "3 ms acq window"

os.chdir(path)

filename = glob.glob('*.dat*')[0]

pixels_peak = np.arange(250, 256, 1)
pixels_noise = np.arange(90, 100, 1)

pixels = np.concatenate((pixels_noise, pixels_peak))

data = f_up.unpack_binary_512(filename)

for i, pixel in enumerate(pixels):
    plt.figure(figsize=(16, 10))
    plt.rcParams.update({'font.size': 22})
    bins = np.arange(0, 4e9, 17.867*1e6)  # bin size of 17.867 us
    plt.hist(data[pixel], bins=bins, color="teal")
    plt.xlabel("Time [ms]")
    plt.ylabel("Counts [-]")
    plt.title("Pixel {}".format(pixel))
    try:
        os.chdir("results/single pixel histograms")
    except Exception:
        os.mkdir("results/single pixel histograms")
        os.chdir("results/single pixel histograms")
    plt.savefig("{file}, pixel {pixel}.png".format(file=filename, pixel=pixel))
    os.chdir("../..")
