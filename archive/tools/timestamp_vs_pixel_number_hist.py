"""Script for plotting a 2D plot of timestamps vs pixel number.

TODO: '-1' data points should be eliminated from the output histogram

"""
import glob
import os

import numpy as np
from matplotlib import pyplot as plt

from functions import unpack as f_up

plt.ioff()

path_3_99 = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"
    "/Data/Ne lamp ext trig/setup 2/FW 2208/3.99 ms"
)

paths = [path_3_99]

pixel_numbers = np.arange(0, 256, 1)

for j, path in enumerate(paths):
    os.chdir(path)
    print("Working in {}.".format(path))

    DATA_FILES = glob.glob("*.dat*")
    for i, file in enumerate(DATA_FILES):
        print("\nWorking on data file {}.".format(file))

        data = f_up.unpack_binary_512(file)
        plt.figure(figsize=(16, 10))
        plt.rcParams.update({"font.size": 22})
        plt.xlabel("Pixel number [-]")
        plt.ylabel("Timestamps [ps]")
        plt.hist(data, bins="auto")
        try:
            os.chdir("results/timestamps vs pixel number")
        except Exception:
            os.mkdir("results/timestamps vs pixel number")
            os.chdir("results/timestamps vs pixel number")
        plt.savefig("{}_hist.png".format(file))
        plt.pause(0.1)
        plt.close()
        os.chdir("../..")
