"""Module for analyzing cross-talk of LinoSPAD2.

A set of functions to calculate and collect the cross-talk data
for the given data sets.

This file can also be imported as a module and contains the following
functions:

    * colect_ct - function for calculating and collecting the cross-talk
    data into a .csv file.

    * plot_ct - function for plotting the cross-talk data from the .csv
    file as the cross-talk vs the distance between the two pixels, for
    which the cross-talk is calculated, in pixels.

"""

import glob
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem

from functions import calc_diff as cd
from functions import unpack as f_up


def collect_ct(
    path,
    pixels,
    board_number: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 10e3,
):
    """Calculate cross-talk and save it to a .csv file.

    Calculate timestamp differences for all pixels in the given range,
    where all timestamp differences are calculated for the first pixel
    in the range. Works with firmware versions "2208" and "2212b". The
    output is saved as a .csv file in the folder with the datafiles.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pixels : array-like
        Array of pixel numbers.
    board_number : str
        The LinoSPAD2 daughterboard number. Only "NL11" and "A5" values
        are accepted.
    fw_ver : str
        Firmware version installed on the LinoSPAD2 motherboard. Only
        "2208" and "2212b" are accepted.
    timestamps : int, optional
        Number of timestamps per pixel per cycle. The default is 512.
    delta_window : float, optional
        A width of a window in which the number of timestamp differences
        are counted. The default value is 10e3 (10ns).

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, either '2208' or '2212b'")

    print("\n> > > Collecting data for cross-talk analysis < < <\n")
    file_name_list = []
    pix1_list = []
    pix2_list = []
    timestamps_list1 = []
    timestamps_list2 = []
    deltas_list = []
    ct_list = []

    os.chdir(path)

    files = glob.glob("*.dat*")
    if fw_ver == "2208":
        for i, file in enumerate(files):
            data = f_up.unpack_numpy(file, board_number, timestamps)
            pix1 = pixels[0]
            for k, pix2 in enumerate(pixels[1:]):
                if pix2 <= pix1:
                    continue

                data_pair = np.vstack((data[pix1], data[pix2]))

                deltas = cd.calc_diff_2208(data_pair, timestamps, delta_window)

                timestamps_pix1 = len(np.where(data[pix1] > 0)[0])
                if timestamps_pix1 == 0:
                    continue
                timestamps_pix2 = len(np.where(data[pix2] > 0)[0])

                ct = len(deltas) * 100 / (timestamps_pix1 + timestamps_pix2)

                file_name_list.append(file)
                pix1_list.append(pix1)
                pix2_list.append(pix2)
                timestamps_list1.append(timestamps_pix1)
                timestamps_list2.append(timestamps_pix2)
                deltas_list.append(len(deltas))
                ct_list.append(ct)

    elif fw_ver == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
        for i, file in enumerate(files):
            data_all = f_up.unpack_2212_numpy(file, board_number, timestamps)

            tdc1, pix_c1 = np.argwhere(pix_coor == pixels[0])[0]
            pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
            data1 = data_all[tdc1].T[1][pix1]

            cycle_ends = np.argwhere(data_all[0].T[0] == -2)
            cycle_ends = np.insert(cycle_ends, 0, 0)
            for j in range(1, len(pixels)):
                if pixels[j] <= pixels[0]:
                    continue

                tdc2, pix_c2 = np.argwhere(pix_coor == pixels[j])[0]
                pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]

                data2 = data_all[tdc2].T[1][pix2]

                deltas = cd.calc_diff_2212(data1, data2, cycle_ends, delta_window)

                timestamps_pix1 = len(np.where(data1 > 0)[0])
                if timestamps_pix1 == 0:
                    continue
                timestamps_pix2 = len(np.where(data2 > 0)[0])

                ct = len(deltas) * 100 / (timestamps_pix1 + timestamps_pix2)

                file_name_list.append(file)
                pix1_list.append(pixels[0])
                pix2_list.append(pixels[j])
                timestamps_list1.append(timestamps_pix1)
                timestamps_list2.append(timestamps_pix2)
                deltas_list.append(len(deltas))
                ct_list.append(ct)

    print("\n> > > Saving data to a .csv file in folder {} < < <\n".format(path))

    dic = {
        "File": file_name_list,
        "Pixel 1": pix1_list,
        "Pixel 2": pix2_list,
        "Timestamps 1": timestamps_list1,
        "Timestamps 2": timestamps_list2,
        "Deltas": deltas_list,
        "CT": ct_list,
    }

    ct_data = pd.DataFrame(dic)

    path_to_save = "C:/Users/bruce/Documents/Quantum astrometry/CT"
    os.chdir(path_to_save)

    if glob.glob("*CT_data_{}-{}.csv*".format(files[0], files[-1])) == []:
        ct_data.to_csv(
            "CT_data_{}-{}.csv".format(files[0], files[-1]),
            index=False,
        )
    else:
        ct_data.to_csv(
            "CT_data_{}-{}.csv".format(files[0], files[-1]),
            mode="a",
            index=False,
            header=False,
        )


def plot_ct(path, pix1, scale: str = "linear"):
    """Plot cross-talk data.

    Parameters
    ----------
    path : str
        Path to the folder where a .csv file with the cross-talk data is
        located.
    pix1 : int
        Pixel number relative to which the cross-talk data should be
        plotted.
    scale : str, optional
        Switch for plot scale, logarithmic or linear. Default is "linear".

    Returns
    -------
    None.

    """
    print("\n> > > Plotting cross-talk vs distance in pixels < < <\n")
    os.chdir(path)

    files = glob.glob("*.dat*")

    os.chdir("C:/Users/bruce/Documents/Quantum astrometry/CT")

    file = glob.glob("*CT_data_{}-{}.csv*".format(files[0], files[-1]))[0]

    plot_name = "{}_{}".format(files[0], files[-1])

    data = pd.read_csv(file)

    distance = []
    ct = []
    yerr = []

    pix1 = pix1

    data_cut = data.loc[data["Pixel 1"] == pix1]

    pix2 = data["Pixel 2"].unique()
    pix2 = np.delete(pix2, np.where(pix2 <= pix1)[0])
    pix2 = np.sort(pix2)

    for i, pix in enumerate(pix2):
        ct_pix = data_cut[data_cut["Pixel 2"] == pix].CT.values

        if ct_pix.size <= 0:
            continue

        distance.append(pix - pix1)
        if len(ct_pix > 1):
            ct.append(np.average(ct_pix))
        else:
            ct.append(ct_pix)
        yerr.append(sem(ct_pix))

        xticks = np.linspace(
            distance[0], distance[-1], int(len(distance) / 10), dtype=int
        )

    plt.rcParams.update({"font.size": 20})

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    if scale == "log":
        plt.yscale("log")
    ax1.errorbar(distance, ct, yerr=yerr, color="salmon")
    ax1.set_xlabel("Distance in pixels [-]")
    ax1.set_ylabel("Average cross-talk [%]")
    ax1.set_title("Pixel {}".format(pix1))
    ax1.set_xticks(xticks)

    plt.savefig("{plot}_{pix}.png".format(plot=plot_name, pix=pix1))

    pickle.dump(fig, open("{}.pickle".format(plot_name), "wb"))
