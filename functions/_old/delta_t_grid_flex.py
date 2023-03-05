"""Script for plotting a grid 5x5 of delta t for different pairs of pixels.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

The output is saved in the `results/delta_t` directory, in the case there is
no such folder, it is created where the data are stored.

This file can also be imported as a module and contains the following
functions:

    * plot_grid - plots a 4x4 grid of delta t for different pairs of pixels for
    5 pixels total

"""

import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up


def plot_grid(path, pix, lines_of_data: int = 512, show_fig: bool = False):
    '''Plots a 4x4 grid of delta t for different pairs of pixels for 5 pixels
    total in the giver range.


    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of 5 pixels for analysis.
    lines_of_data : int, optional
        Number of data points per pixel per acquisition cycle. The default is
        512.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.

    Returns
    -------
    None.

    '''

    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    for num, filename in enumerate(DATA_FILES):
        print("================================\n"
              "Working on {}\n"
              "================================".format(filename))
        data = f_up.unpack_binary_flex(filename, lines_of_data)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]

        # check if the figure should appear in a separate window or not at all
        if show_fig is True:
            plt.ion()
        else:
            plt.ioff()

        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(len(pix)-1, len(pix)-1, figsize=(24, 24))

        y_max_all = 0

        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue

                data_pair = np.vstack((data_pix[q], data_pix[w]))

                minuend = len(data_pair)
                timestamps_total = len(data_pair[0])
                subtrahend = len(data_pair)
                timestamps = lines_of_data

                output = []

                for i in tqdm(range(minuend)):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps_total):
                        if j % lines_of_data == 0:
                            acq = acq + 1  # next acq cycle
                        if data_pair[i][j] == -1:
                            continue
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, 53-45
                            for p in range(timestamps):
                                n = lines_of_data*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif np.abs(data_pair[i][j]
                                            - data_pair[k][n]) > 2.5e3:
                                    continue
                                elif np.abs(data_pair[i][j]
                                            - data_pair[k][n]) < -2.5e3:
                                    continue
                                else:
                                    output.append(data_pair[i][j]
                                                  - data_pair[k][n])

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"

                try:
                    bins = np.arange(np.min(output), np.max(output),
                                     17.857*2)
                except Exception:
                    continue
                axs[q][w-1].set_xlabel('\u0394t [ps]')
                axs[q][w-1].set_ylabel('Timestamps [-]')
                n, b, p = axs[q][w-1].hist(output, bins=bins,
                                           color=chosen_color)
                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2,
                                     ".2f")
                except Exception:
                    arg_max = None
                    pass

                y_max = np.max(n)
                if y_max_all < y_max:
                    y_max_all = y_max

                axs[q][w-1].set_ylim(0, y_max+10)
                axs[q][w-1].set_xlim(-2.5e3, 2.5e3)

                axs[q][w-1].set_title('Pixels {p1}-{p2}\nPeak position {pp}'
                                      .format(p1=pix[q],
                                              p2=pix[w],
                                              pp=arg_max))

        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue
                axs[q][w-1].set_ylim(0, y_max_all+10)

        try:
            os.chdir("results/delta_t")
        except Exception:
            os.mkdir("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid.png".format(name=filename))
        os.chdir("../..")
