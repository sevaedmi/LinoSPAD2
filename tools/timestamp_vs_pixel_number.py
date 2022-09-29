"""Script for plotting a 2D plot of timestamps vs pixel number.

TODO: add control if the result already exists.

"""
from functions import unpack as f_up
import glob
import os
from matplotlib import pyplot as plt

plt.ioff()

path_3_99 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/Ne lamp ext trig/setup 2/FW 2208/3.99 ms"

paths = [path_3_99]

for j, path in enumerate(paths):
    os.chdir(path)
    print("Working in {}.".format(path))

    DATA_FILES = glob.glob('*.dat*')
    for i, file in enumerate(DATA_FILES):
        print("\nWorking on data file {}.".format(file))

        data = f_up.unpack_binary_512(file)
        plt.figure(figsize=(16, 10))
        plt.rcParams.update({'font.size': 22})
        plt.xlabel('Pixel number [-]')
        plt.ylabel('Timestamps [ps]')
        plt.plot(data, '.')
        try:
            os.chdir("results/timestamps vs pixel number")
        except Exception:
            os.mkdir("results/timestamps vs pixel number")
            os.chdir("results/timestamps vs pixel number")
        plt.savefig("{}.png".format(file))
        plt.pause(0.1)
        plt.close()
        os.chdir("../..")
