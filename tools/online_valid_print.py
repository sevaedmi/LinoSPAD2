"""Script for online printing of histograms of valid timestamps vs pixel
number. Waits for a new data file, processes it when the file appears, shows
and saves the output figure. Then the cycle repeats.

"""

import glob
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from functions import unpack as f_up

# =============================================================================
# Looking for last created file
# =============================================================================

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
        "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
        "binary/Ne lamp ext trig/setup 2/3 ms acq window"

path_save = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
        "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
        "binary/Ne lamp ext trig/setup 2/3 ms acq window/online"

os.chdir(path)

last_file_ctime = 0
plt.ion()

while True:

    DATA_FILES = glob.glob('*.dat*')
    last_file = max(DATA_FILES, key=os.path.getctime)

    new_file_ctime = os.path.getctime(last_file)

    if new_file_ctime <= last_file_ctime:
        print("Waiting for new file.")
        last_file_ctime = new_file_ctime
        time.sleep(10)
        continue

# =============================================================================
# Data analysis
# =============================================================================
    last_file_ctime = new_file_ctime

    print("Analysing the last file.")
    data = f_up.unpack_binary_flex(last_file, lines_of_data=512)

    valid_per_pixel = np.zeros(256)

    for j in range(len(data)):
        valid_per_pixel[j] = len(np.where(data[j] > 0)[0])

    peak = np.max(valid_per_pixel[200:])

    plt.close('all')
    plt.pause(1)
    plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 20})
    plt.title("Peak is {}".format(peak))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Valid timestamps [-]")
    plt.yscale('log')
    plt.plot(valid_per_pixel, 'o')
    plt.show()

    plt.pause(1)

    os.chdir(path_save)
    plt.savefig("{}.png".format(last_file))
    os.chdir('..')
