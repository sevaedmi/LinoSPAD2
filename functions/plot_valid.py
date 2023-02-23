""" Module with scripts for plotting the LinoSPAD2 output, namely the number
of timestamps in each pixel.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * plot_pixel_hist - plots a histogram of timestamps for a single pixel.
    The function can be used mainly for controlling the homogenity of the
    LinoSPAD2 output.

    * plot_valid_df - plots the number of valid timestamps vs the pixel number;
    works with tidy dataframes. Currently, the fastest option for plotting valid
    timestamps.

    * plot_calib - plots the number of valid timestamps vs the pixel number, using
    the calibration data. Imputing the board number is required.

"""

import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from functions import unpack as f_up


def plot_pixel_hist(path, pix1, timestamps: int = 512, show_fig: bool = False):
    """
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

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()
    for i, num in enumerate(DATA_FILES):
        print(
            "=====================================================\n"
            "Plotting pixel histograms, Working on {}\n"
            "=====================================================".format(num)
        )

        data = f_up.unpack_numpy(num, timestamps)

        if pix1 is None:
            pixels = np.arange(145, 165, 1)
        else:
            pixels = pix1
        for i, pixel in enumerate(pixels):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({"font.size": 22})
            bins = np.arange(0, 4e9, 17.867 * 1e6)  # bin size of 17.867 us
            plt.hist(data[pixel], bins=bins, color="teal")
            plt.xlabel("Time [ms]")
            plt.ylabel("Counts [-]")
            plt.title("Pixel {}".format(pixel))
            try:
                os.chdir("results/single pixel histograms")
            except Exception:
                os.makedirs("results/single pixel histograms")
                os.chdir("results/single pixel histograms")
            plt.savefig("{file}, pixel {pixel}.png".format(file=num, pixel=pixel))
            os.chdir("../..")


def plot_valid(
    path,
    board_number,
    timestamps: int = 512,
    mask: list = [],
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
):
    """
    Function for plotting the number of valid timestamps per pixel. Uses
    the calibration data.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    board_number : str
        The LinoSPAD2 board number. Required for choosing the correct
        calibration data.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. Default is "512".
    mask : array-like
        Array of pixels indices to mask.
    scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

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

        data_matrix = f_up.unpack_numpy(num, board_number, timestamps)

        for j in range(len(data_matrix)):
            valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

        valid_per_pixel[mask] = 0

        peak = np.max(valid_per_pixel)

        if "Ne" and "540" in path:
            chosen_color = "seagreen"
        elif "Ne" and "656" in path:
            chosen_color = "orangered"
        elif "Ne" and "585" in path:
            chosen_color = "goldenrod"
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


def plot_valid_mult(
    path,
    board_number,
    timestamps: int = 512,
    mask: list = [],
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    mult_files: bool = False,
):
    os.chdir(path)

    valid_per_pixel = np.zeros(256)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    if mult_files is True:
        # os.chdir(path)
        # if len(glob.glob(".dat")) > 10:
        #     print("Too many files.")
        #     sys.exit()
        print(
            "=================================================\n"
            "Plotting valid timestamps, Working in {}\n"
            "=================================================".format(path)
        )
        data, plot_name = f_up.unpack_mult(path, board_number, timestamps)

    else:
        os.chdir(path)
        files = glob.glob("*.dat*")
        last_file = max(files, key=os.path.getctime)
        print(
            "=================================================\n"
            "Plotting valid timestamps, Working on {}\n"
            "=================================================".format(last_file)
        )
        data = f_up.unpack_numpy(last_file, board_number, timestamps)

    for j in range(len(data)):
        valid_per_pixel[j] = len(np.where(data[j] > 0)[0])

    valid_per_pixel[mask] = 0

    peak = int(np.max(valid_per_pixel))

    if "Ne" and "540" in path:
        chosen_color = "seagreen"
    elif "Ne" and "656" in path:
        chosen_color = "orangered"
    elif "Ne" and "585" in path:
        chosen_color = "goldenrod"
    elif "Ar" in path:
        chosen_color = "mediumslateblue"
    else:
        chosen_color = "salmon"
    plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 20})
    plt.title("Peak is {peak}".format(peak=peak))
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
    if mult_files is True:
        plt.savefig("{name}.png".format(name=plot_name))
    else:
        plt.savefig("{name}.png".format(name=last_file))
    plt.pause(0.1)
    if show_fig is False:
        plt.close("all")
    os.chdir("..")
