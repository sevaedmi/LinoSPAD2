"""Script for plotting a 2D plot of timestamps vs pixel number.

TODO: add control if the result already exists.

"""
from functions import unpack as f_up
import glob
import os
from matplotlib import pyplot as plt

plt.ioff()

# path_1 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
#     "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
#     "1 ms acq window"

# path_2 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
#     "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
#     "2 ms acq window"

# path_3 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
#     "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
#     "3 ms acq window"

# path_3_9 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
#     "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
#     "3.9 ms acq window"

# path_0_500 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
#     "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
#     "500 us acq window"

path_3_99 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
    "3.99 ms acq window"

# paths = [path_1, path_2, path_3, path_3_9, path_0_500]
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
        # plt.pause(1)
        # plt.close('all')
        os.chdir("../..")
