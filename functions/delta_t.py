""" Module with scripts for calculating and plotting the timestamp differences
for different pairs of pixels.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * plot_delta_separate - plots delta t for each pair of pixels in
    the range of 251-255
    * plot_grid - plots a 4x4 grid of delta t for different pairs of pixels for
    5 pixels total
    * plot_delta_separate - plots delta t for each pair of pixels in
    the range of 251-255

"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up


def plot_grid(path, pix, timestamps: int = 512, show_fig: bool = False):
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

        print("================================\n"
              "Working on {}\n"
              "================================".format(filename))

        data = f_up.unpack_binary_flex(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]

        plt.rcParams.update({'font.size': 22})
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


def plot_delta_auto_zoom(path, pix, timestamps: int = 512):
    '''
    Plots delta t for each pair of pixels in the given range. The plots are
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

        print("================================\n"
              "Working on {}\n"
              "================================".format(filename))

        data = f_up.unpack_binary_flex(filename, timestamps)

        data_1 = data[pix[0]]  # 1st pixel
        data_2 = data[pix[1]]  # 2nd pixel
        data_3 = data[pix[2]]  # 3d pixel
        data_4 = data[pix[3]]  # 4th pixel
        data_5 = data[pix[4]]  # 5th pixel

        pixel_numbers = np.arange(pix[0], pix[-1]+1, 1)

        all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))

        plt.rcParams.update({'font.size': 22})
        plt.ioff()

        for q in range(5):
            for w in range(5):
                if w <= q:
                    continue

                data_pair = np.vstack((all_data[q], all_data[w]))

                minuend = len(data_pair)-1  # i=255
                timestamps = len(data_pair[0])
                subtrahend = len(data_pair)  # k=254
                timestamps = 512  # lines of data in the acq cycle

                output = []

                for i in tqdm(range(minuend)):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps):
                        if data_pair[i][j] == -1:
                            continue
                        if j % timestamps == 0:
                            acq = acq + 1  # next acq cycle
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, etc.
                            for p in range(timestamps):
                                n = timestamps*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] > 1e4:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] < -1e4:
                                    continue
                                else:
                                    output.append(data_pair[i][j]
                                                  - data_pair[k][n])
                try:
                    bins = np.arange(np.min(output), np.max(output), 17.857*50)
                except Exception:
                    continue

                n, b, p = plt.hist(output, bins=bins)
                plt.close()

                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = (bins[n_max] + bins[n_max + 1]) / 2
                except Exception:
                    arg_max = None
                    pass

                output = []

                if arg_max is None:
                    continue

                for i in tqdm(range(minuend), desc='Working on the zoom'):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps):
                        if data_pair[i][j] == -1:
                            continue
                        if j % timestamps == 0:
                            acq = acq + 1  # next acq cycle
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, etc.
                            for p in range(timestamps):
                                n = timestamps*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif (data_pair[i][j] - data_pair[k][n]
                                      > (int(arg_max) + 1000)):
                                    continue
                                elif (data_pair[i][j] - data_pair[k][n]
                                      < (int(arg_max) - 1000)):
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

                plt.figure(figsize=(16, 10))
                plt.xlabel('\u0394t [ps]')
                plt.ylabel('Timestamps [-]')
                n, b, p = plt.hist(output, bins=bins, color=chosen_color)

                arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")

                plt.title('{filename}\nPeak position: {peak}\nPixels {p1}-{p2}'
                          .format(filename=filename, peak=arg_max,
                                  p1=pixel_numbers[q], p2=pixel_numbers[w]))

                try:
                    os.chdir("results/delta_t/zoom")
                except Exception:
                    os.mkdir("results/delta_t/zoom")
                    os.chdir("results/delta_t/zoom")
                plt.savefig("{name}_pixels {p1}-{p2}.png"
                            .format(name=filename, p1=pixel_numbers[q],
                                    p2=pixel_numbers[w]))
                plt.pause(0.1)
                plt.close()
                os.chdir("../../..")


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

        print("================================\n"
              "Working on {}\n"
              "================================".format(filename))

        data = f_up.unpack_binary_flex(filename, timestamps)

        filename = glob.glob('*.dat*')[0]

        lod = timestamps

        data = f_up.unpack_binary_flex(filename, lod)

        data_1 = data[pix[0]]  # 1st pixel
        data_2 = data[pix[1]]  # 2nd pixel
        data_3 = data[pix[2]]  # 3d pixel
        data_4 = data[pix[3]]  # 4th pixel
        data_5 = data[pix[4]]  # 5th pixel

        pixel_numbers = np.arange(pix[0], pix[-1]+1, 1)

        all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))

        plt.rcParams.update({'font.size': 22})

        for q in range(5):
            for w in range(5):
                if w <= q:
                    continue

                data_pair = np.vstack((all_data[q], all_data[w]))

                minuend = len(data_pair)-1  # i=255
                timestamps = len(data_pair[0])
                subtrahend = len(data_pair)  # k=254

                output = []

                for i in tqdm(range(minuend)):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps):
                        if data_pair[i][j] == -1:
                            continue
                        if j % 512 == 0:
                            acq = acq + 1  # next acq cycle
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, etc.
                            for p in range(timestamps):
                                n = 512*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] > 1.5e3:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] < -1.5e3:
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

                plt.figure(figsize=(16, 10))
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
                          .format(filename=filename, peak=arg_max,
                                  p1=pixel_numbers[q], p2=pixel_numbers[w]))

                try:
                    os.chdir("results/delta_t/zoom")
                except Exception:
                    os.mkdir("results/delta_t/zoom")
                    os.chdir("results/delta_t/zoom")
                plt.savefig("{name}_pixels {p1}-{p2}.png"
                            .format(name=filename, p1=pixel_numbers[q],
                                    p2=pixel_numbers[w]))
                plt.pause(0.1)
                plt.close()
                os.chdir("../../..")
