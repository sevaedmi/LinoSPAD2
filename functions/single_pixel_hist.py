"""Script for plotting histograms of timestamps for single pixels in the
range 250-255 (illuminated part), and 90-100 (noisy part).

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

The output is saved in the `results/single pixel histograms` directory,
in the case there is no such folder, it is created where the data are stored.

This file can also be imported as a module and contains the following
functions:

    * plot_pixel_hist - plot separate histograms for a range of pixels

"""

import glob
import os
import numpy as np
from functions import unpack as f_up
from matplotlib import pyplot as plt


def plot_pixel_hist(path, show_fig: bool = False):
    '''Plots a histogram for each pixel in a preset range.


    Parameters
    ----------
    path : str
        Path to data file.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.

    Returns
    -------
    None.

    '''

    os.chdir(path)

    filename = glob.glob('*.dat*')[0]

    pixels_peak = np.arange(250, 256, 1)
    pixels_noise = np.arange(90, 100, 1)

    pixels = np.concatenate((pixels_noise, pixels_peak))

    data = f_up.unpack_binary_512(filename)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

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
        plt.savefig("{file}, pixel {pixel}.png".format(file=filename,
                                                       pixel=pixel))
        os.chdir("../..")
