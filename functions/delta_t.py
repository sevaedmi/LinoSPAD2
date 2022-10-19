""" Module with scripts for calculating and plotting the timestamp differences
for different pairs of pixels.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * plot_grid - function for plotting a grid of NxN plots (N for number of
      pixels) of timestamp differences

    * plot_delta_separate - function for plotting separate figures of
    timestamp differences for each pair of pixels in the given range

"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up
from functions.calc_diff import calculate_differences as cd


def plot_grid(
    path, pix, timestamps: int = 512, show_fig: bool = False, same_y: bool = True
):
    """
    Plots a grid of delta t for different pairs of pixels for the
    pixels in the given range. The output is saved in the "results/delta_t"
    folder. In the case the folder does not exist, it is created automatically.


    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of pixels for analysis.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    same_y : bool, optional
        Switch for setting the same ylim for all plots in the grid. The
        default is True.

    Returns
    -------
    None.

    """

    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()
    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):

        print(
            "=====================================================\n"
            "Plotting a delta t grid, Working on {}\n"
            "=====================================================\n".format(filename)
        )

        data = f_up.unpack_binary_flex(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num1 in enumerate(pix):
            data_pix[i] = data[num1]
        plt.rcParams.update({"font.size": 22})
        fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(24, 24))

        # check if the y limits of all plots should be the same
        if same_y is True:
            y_max_all = 0
        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                data_pair = np.vstack((data_pix[q], data_pix[w]))

                delta_ts = cd(
                    data_pair,
                    timestamps=timestamps,
                    range_left=-2.5e6,
                    range_right=2.5e6,
                )

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"
                try:
                    bins = np.arange(np.min(delta_ts), np.max(delta_ts), 17.857 * 28e2)
                except Exception:
                    print("Couldn't calculate bins: probably not enough delta ts.")
                    continue
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("Timestamps [-]")
                n, b, p = axs[q][w - 1].hist(delta_ts, bins=bins, color=chosen_color)
                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
                except Exception:
                    arg_max = None
                if same_y is True:
                    try:
                        y_max = np.max(n)
                    except ValueError:
                        y_max = 0
                        print("\nCould not find maximum y value\n")
                    if y_max_all < y_max:
                        y_max_all = y_max
                    axs[q][w - 1].set_ylim(0, y_max + 4)
                axs[q][w - 1].set_xlim(-2.5e6, 2.5e6)

                axs[q][w - 1].set_title(
                    "Pixels {p1}-{p2}\nPeak position {pp}".format(
                        p1=pix[q], p2=pix[w], pp=arg_max
                    )
                )
        if same_y is True:
            for q in range(len(pix)):
                for w in range(len(pix)):
                    if w <= q:
                        continue
                    axs[q][w - 1].set_ylim(0, y_max_all + 10)
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.mkdir("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid.png".format(name=filename))
        os.chdir("../..")


def plot_delta_separate(path, pix, timestamps: int = 512):
    """
    Plots delta t for each pair of pixels in the given range.  The plots are
    saved in the "results/delta_t/zoom" folder. In the case the folder does
    not exist, it is created automatically.

    Parameters
    ----------
    path : str
        Path to data file.
    pix : array-like
        Array of indices of 5 pixels for analysis.
    timestamps : int
        Number of timestamps per acq cycle per pixel in the file. The default
        is 512.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):

        print(
            "======================================================\n"
            "Plotting timestamp differences, Working on {}\n"
            "======================================================".format(filename)
        )

        data = f_up.unpack_binary_flex(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num1 in enumerate(pix):
            data_pix[i] = data[num1]
        plt.rcParams.update({"font.size": 22})

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                data_pair = np.vstack((data_pix[q], data_pix[w]))

                delta_ts = cd(data_pair, timestamps=timestamps)

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"
                try:
                    bins = np.arange(np.min(delta_ts), np.max(delta_ts), 17.857 * 2)
                except Exception:
                    continue
                plt.figure(figsize=(11, 7))
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                (n,) = plt.hist(delta_ts, bins=bins, color=chosen_color)

                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
                except Exception:
                    arg_max = None
                plt.title(
                    "{filename}\nPeak position: {peak}\nPixels {p1}-{p2}".format(
                        filename=filename, peak=arg_max, p1=pix[q], p2=pix[w]
                    )
                )

                try:
                    os.chdir("results/delta_t/zoom")
                except Exception:
                    os.mkdir("results/delta_t/zoom")
                    os.chdir("results/delta_t/zoom")
                plt.savefig(
                    "{name}_pixels {p1}-{p2}.png".format(
                        name=filename, p1=pix[q], p2=pix[w]
                    )
                )
                plt.pause(0.1)
                plt.close()
                os.chdir("../../..")
