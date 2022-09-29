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


def online_plot_valid(path, pix_range, lines_of_data: int = 512):
    '''Real-time plotting of number of valid timestamps from the last data
    file. Waits for a new file then analyzes it and shows the plot.

    Parameters
    ----------
    path : str
        Path to where the data files are.
    pix_range : array-like
        Range of pixels for which the maximum value is showed in the plot.
    lines_of_data : int, optional
        Number of data points per pixel per acquisition cycle. The default is
        512.

    Returns
    -------
    None.

    '''
    # =========================================================================
    # Looking for last created file
    # =========================================================================

    path_save = path + "/results/online"

    os.chdir(path)

    last_file_ctime = 0
    plt.ion()

    while True:

        DATA_FILES = glob.glob('*.dat*')
        try:
            last_file = max(DATA_FILES, key=os.path.getctime)
        except Exception:
            print("Waiting for a file")
            time.sleep(5)
            continue

        new_file_ctime = os.path.getctime(last_file)

        if new_file_ctime <= last_file_ctime:
            print("Waiting for new file.")
            last_file_ctime = new_file_ctime
            time.sleep(5)
            continue

    # =========================================================================
    # Data analysis
    # =========================================================================
        last_file_ctime = new_file_ctime

        print("Analysing the last file.")
        data = f_up.unpack_binary_flex(last_file, lines_of_data)

        valid_per_pixel = np.zeros(256)

        for j in range(len(data)):
            valid_per_pixel[j] = len(np.where(data[j] > 0)[0])

        peak = np.max(valid_per_pixel[pix_range])

        if "Ne" and "540" in path:
            chosen_color = "seagreen"
        elif "Ne" and "656" in path:
            chosen_color = "orangered"
        elif "Ar" in path:
            chosen_color = "mediumslateblue"
        else:
            chosen_color = "salmon"

        plt.close('all')
        plt.pause(1)
        plt.figure(figsize=(11, 7))
        plt.rcParams.update({"font.size": 20})
        plt.title("Peak is {}".format(peak))
        plt.xlabel("Pixel [-]")
        plt.ylabel("Valid timestamps [-]")
        plt.yscale('log')
        plt.plot(valid_per_pixel, 'o', color=chosen_color)
        plt.show()

        plt.pause(1)
        try:
            os.chdir(path_save)
        except Exception:
            os.mkdir(path_save)
            os.chdir(path_save)
        plt.savefig("{}.png".format(last_file))
        os.chdir('../..')
