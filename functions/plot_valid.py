""" Module with scripts for plotting the LinoSPAD2 output, namely the number
of timestamps in each pixel.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * plot_valid - plots the number of valid timestamps as a function of
    pixel number
    * online_plot_valid - in the chosen folder, looks for the last data file
    created and plots the number of valid timestamps as a functions of
    pixel number, waits for the next file and repeats the cycle
    * plot_pixel_hist - plots a histogram of timestamps for a single pixel.
    The function can be used mainly for controlling the homogenity of the
    LinoSPAD2 output.

"""

import glob
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from functions import unpack as f_up


def plot_valid(path, pix, timestamps, scale: str = 'linear',
               show_fig: bool = False):
    '''
    Plots number of valid timestamps in each pixel for each 'dat' file in
    given folder. The plots are saved as 'png' in the 'results' folder. In
    the case there is no such folder, it is created where the data files are.

    Parameters
    ----------
    path : str
        Location of the 'dat' files with the data from LinoSPAD2.
    pix : array-like
        Array of indices of 5 pixels for analysis.
    timestamps : int
        Number of timestamps per acquistion cycle per pixel.
    scale : str, optional
        Use 'log' for logarithmic scale, leave empty for linear. Default is
        'linear'.

    Returns
    -------
    None.

    '''
    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    valid_per_pixel = np.zeros(256)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    for i, num in enumerate(DATA_FILES):

        print("=========================================\n"
              "Plotting timestamps, Working on {}\n"
              "=========================================".format(num))

        data_matrix = f_up.unpack_binary_flex(num, timestamps)

        for j in range(len(data_matrix)):
            valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

        peak = np.max(valid_per_pixel[pix[0]:pix[-1]])

        if "Ne" and "540" in path:
            chosen_color = "seagreen"
        elif "Ne" and "656" in path:
            chosen_color = "orangered"
        elif "Ar" in path:
            chosen_color = "mediumslateblue"
        else:
            chosen_color = "salmon"

        plt.figure(figsize=(16, 10))
        plt.rcParams.update({"font.size": 20})
        plt.title("{file}\n Peak is {peak}".format(file=num, peak=peak))
        plt.xlabel("Pixel [-]")
        plt.ylabel("Valid timestamps [-]")
        if scale == 'log':
            plt.yscale('log')
        plt.plot(valid_per_pixel, 'o', color=chosen_color)

        try:
            os.chdir("results")
        except Exception:
            os.mkdir("results")
            os.chdir("results")

        plt.savefig("{}.png".format(num))
        plt.pause(0.1)
        if show_fig is False:
            plt.close('all')
        os.chdir("..")


def online_plot_valid(path, pix_range, timestamps: int = 512):
    '''
    Real-time plotting of number of valid timestamps from the last data
    file. Waits for a new file then analyzes it and shows the plot. The
    output is saved in the "results/online" directory. In the case the folder
    does not exist, it is created.

    Parameters
    ----------
    path : str
        Path to where the data files are.
    pix_range : array-like
        Range of pixels for which the maximum is calculated and is then shown
        in the plot.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
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

    print("=============================\n"
          "Online plotting of timestamps\n"
          "=============================")
    while True:

        try:
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
            data = f_up.unpack_binary_flex(last_file, timestamps)

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
        except KeyboardInterrupt:
            break


def plot_pixel_hist(path, pix1, timestamps: int = 512, show_fig: bool = False):
    '''
    Plots a histogram for each pixel in a preset range.

    Parameters
    ----------
    path : str
        Path to data file.
    pix1 : array-like
        Array of pixels indices. Preferably pixels where the peak is.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.

    Returns
    -------
    None.

    '''

    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    for i, num in enumerate(DATA_FILES):

        print("========================================\n"
              "Plotting pixel histograms, Working on {}\n"
              "=========================================".format(num))

        data = f_up.unpack_binary_flex(num, timestamps)

        if pix1 is None:
            pixels = np.arange(145, 165, 1)
        else:
            pixels = pix1

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
            plt.savefig("{file}, pixel {pixel}.png".format(file=num,
                                                           pixel=pixel))
            os.chdir("../..")
