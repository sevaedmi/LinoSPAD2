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

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
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

        data_1 = data[pix[0]]  # 1st pixel
        data_2 = data[pix[1]]  # 2nd pixel
        data_3 = data[pix[2]]  # 3d pixel
        data_4 = data[pix[3]]  # 4th pixel
        data_5 = data[pix[4]]  # 5th pixel

        pixel_numbers = np.arange(pix[0], pix[-1]+1, 1)

        all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))

        # check if the figure should appear in a separate window or not at all
        if show_fig is True:
            plt.ion()
        else:
            plt.ioff()

        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(4, 4, figsize=(24, 24))

        y_max_all = 0

        for q in range(5):
            for w in range(5):
                if w <= q:
                    continue

                data_pair = np.vstack((all_data[q], all_data[w]))

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
                                      .format(p1=pixel_numbers[q],
                                              p2=pixel_numbers[w],
                                              pp=arg_max))

        for q in range(5):
            for w in range(5):
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


def plot_peak_vs_peak(path, pix1, pix2, timestamps: int = 512):
    """
    Function for calculating timestamp differences between two groups of pixels where
    the light beam falls. The differences are plotted as a histogram.

    Parameters
    ----------
    path : str
        Path to data files.
    pix1 : list or array-like
        List of pixel numbers from the first peak.
    pix2 : list or array-like
        List of pixel numbers from the second peak.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, file in enumerate(DATA_FILES):
        data = f_up.unpack_binary_df(file, timestamps)

        deltas_total = pd.DataFrame()

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix1)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix2)), desc="Subtrahend pixel"):
                deltas = calc_diff_df(
                    data[data.Pixel == pix1[q]], data[data.Pixel == pix2[w]]
                )
                deltas_total = pd.concat([deltas_total, deltas], ignore_index=True)
        if "Ne" and "540" in path:
            chosen_color = "seagreen"
        elif "Ne" and "656" in path:
            chosen_color = "orangered"
        elif "Ar" in path:
            chosen_color = "mediumslateblue"
        else:
            chosen_color = "salmon"
        try:
            bins = np.arange(
                int(deltas_total.min()), int(deltas_total.max()), 17.857 * 2
            )
        except Exception:
            continue
        fig_sns = sns.histplot(
            x="Delta t", data=deltas_total, color=chosen_color, bins=bins
        )
        fig = fig_sns.get_figure()
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.makedirs("results/delta_t")
            os.chdir("results/delta_t")
        fig.savefig("{name}_peak_v_peak_df.png".format(name=file))
        os.chdir("../..")