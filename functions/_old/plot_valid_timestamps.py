"""Plots number of valid timestamps in each pixel for each 'dat' file in the
folder.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

The output is saved in the `results` directory, in the case there is no such
folder, it is created where the data are stored.

This file can also be imported as a module and contains the following
functions:

    * plot_valid_per_pixel - plot number of valid timestamps in each pixel

"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

# =============================================================================
# Data collected with the sensor cover but without the optical fiber attached
# =============================================================================


def plot_valid_per_pixel(path, pix, lod, scale: str = "linear", show_fig: bool = False):
    """Plots number of valid timestamps in each pixel for each 'dat' file in
    given folder. The plots are saved as 'png' in the 'results' folder. In
    the case there is no such folder, it is created where the data files are.

    Parameters
    ----------
    path : str
        Location of the 'dat' files with the data from LinoSPAD2.
    pix : array-like
        Array of indices of 5 pixels for analysis.
    lod : int
        Lines of data per acquistion cycle.
    scale : str
        Use 'log' for logarithmic scale, leave empty for linear.

    Returns
    -------
    None.

    """
    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    valid_per_pixel = np.zeros(256)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    for i, num in enumerate(DATA_FILES):
        print(
            "================================\n"
            "Working on {}\n"
            "================================".format(num)
        )
        data_matrix = f_up.unpack_binary_flex(num, lod)
        for j in range(len(data_matrix)):
            valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

        peak = np.max(valid_per_pixel[pix[0] : pix[-1]])

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
        if scale == "log":
            plt.yscale("log")
        plt.plot(valid_per_pixel, "o", color=chosen_color)

        try:
            os.chdir("results")
        except Exception:
            os.mkdir("results")
            os.chdir("results")

        plt.savefig("{}.png".format(num))
        plt.pause(0.1)
        if show_fig is False:
            plt.close("all")
        os.chdir("..")


def plot_valid(
    path,
    pix,
    timestamps,
    mask: list = [],
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
):
    """
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
    mask : list, optional,
        A list of pixel indices. If provided, these pixels will be cut out.
        Default is "[]".
    scale : str, optional
        Use 'log' for logarithmic scale, leave empty for linear. Default is
        'linear'.
    style : str, optional
        What style for the plot should be used. Default is "-o".
    show_fig : bool, optional
        Switch for showing the plot. Default is "False".

    Returns
    -------
    None.

    """
    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    valid_per_pixel = np.zeros(256)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()
    for i, num in enumerate(DATA_FILES):
        print(
            "=================================================\n"
            "Plotting timestamps, Working on {}\n"
            "=================================================".format(num)
        )

        data_matrix = f_up.unpack_numpy(num, timestamps)

        for j in range(len(data_matrix)):
            valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])
        peak = np.max(valid_per_pixel[pix[0] : pix[-1]])

        valid_per_pixel[mask] = 0

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
        if scale == "log":
            plt.yscale("log")
        plt.plot(valid_per_pixel, style, color=chosen_color)

        try:
            os.chdir("results")
        except Exception:
            os.makedirs("results")
            os.chdir("results")
        plt.savefig("{}.png".format(num))
        plt.pause(0.1)
        if show_fig is False:
            plt.close("all")
        os.chdir("..")
