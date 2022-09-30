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


def plot_grid(path, pix, timestamps: int = 512, show_fig: bool = False,
              same_y: bool = True):
    '''
    Plots a grid of delta t for different pairs of pixels for the
    pixels in the given range. The output is saved in the "results/delta_t"
    folder. In the case the folder does not exist, it is created automatically.


    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of 5 pixels for analysis.
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

    '''

    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    for num, filename in enumerate(DATA_FILES):

        print("=====================================================\n"
              "Plotting a delta t grid, Working on {}\n"
              "====================================================="
              .format(filename))

        data = f_up.unpack_binary_flex(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]

        plt.rcParams.update({'font.size': 22})
        fig, axs = plt.subplots(len(pix)-1, len(pix)-1, figsize=(24, 24))

        if same_y is True:
            y_max_all = 0

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc='Minuend pixel   '):
            for w in tqdm(range(len(pix)), desc='Subtrahend pixel'):
                if w <= q:
                    continue

                data_pair = np.vstack((data_pix[q], data_pix[w]))

                minuend = len(data_pair)
                timestamps_total = len(data_pair[0])
                subtrahend = len(data_pair)

                output = []

                for i in range(minuend):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps_total):
                        if j % timestamps == 0:
                            acq = acq + 1  # next acq cycle
                        if data_pair[i][j] == -1:
                            continue
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, 53-45
                            for p in range(timestamps):
                                n = timestamps*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif (data_pair[i][j] - data_pair[k][n]
                                      > 2.5e3):
                                    continue
                                elif (data_pair[i][j] - data_pair[k][n]
                                      < -2.5e3):
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

                if same_y is True:
                    try:
                        y_max = np.max(n)
                    except ValueError:
                        print("\nCould not find maximum y value\n")
                        pass

                    if y_max_all < y_max:
                        y_max_all = y_max

                    axs[q][w-1].set_ylim(0, y_max+4)
                axs[q][w-1].set_xlim(-2.5e3, 2.5e3)

                axs[q][w-1].set_title('Pixels {p1}-{p2}\nPeak position {pp}'
                                      .format(p1=pix[q],
                                              p2=pix[w],
                                              pp=arg_max))

        if same_y is True:
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


def plot_delta_separate(path, pix, timestamps: int = 512):
    '''
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

    '''

    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    for num, filename in enumerate(DATA_FILES):

        print("======================================================\n"
              "Plotting timestamp differences, Working on {}\n"
              "======================================================"
              .format(filename))

        data = f_up.unpack_binary_flex(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]

        plt.rcParams.update({'font.size': 22})

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc='Minuend pixel   '):
            for w in tqdm(range(len(pix)), desc='Subtrahend pixel'):
                if w <= q:
                    continue

                data_pair = np.vstack((data_pix[q], data_pix[w]))

                minuend = len(data_pair)-1  # i=255
                subtrahend = len(data_pair)  # k=254
                timestamps_total = len(data_pair[0])

                output = []

                for i in tqdm(range(minuend)):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps_total):
                        if j % timestamps == 0:
                            acq = acq + 1  # next acq cycle
                        if data_pair[i][j] == -1:
                            continue
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, etc.
                            for p in range(timestamps):
                                n = timestamps*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] > 2.5e3:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] < -2.5e3:
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
                    bins = np.arange(np.min(output), np.max(output), 17.857*2)
                except Exception:
                    continue

                plt.figure(figsize=(11, 7))
                plt.xlabel('\u0394t [ps]')
                plt.ylabel('Timestamps [-]')
                n, b, p = plt.hist(output, bins=bins, color=chosen_color)

                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2,
                                     ".2f")
                except Exception:
                    arg_max = None
                    pass

                plt.title('{filename}\nPeak position: {peak}\nPixels {p1}-{p2}'
                          .format(filename=filename, peak=arg_max, p1=pix[q],
                                  p2=pix[w]))

                try:
                    os.chdir("results/delta_t/zoom")
                except Exception:
                    os.mkdir("results/delta_t/zoom")
                    os.chdir("results/delta_t/zoom")
                plt.savefig("{name}_pixels {p1}-{p2}.png"
                            .format(name=filename, p1=pix[q], p2=pix[w]))
                plt.pause(0.1)
                plt.close()
                os.chdir("../../..")
