"""Module for analyzing cross-talk of LinoSPAD2.

A set of functions to calculate and collect the cross-talk data
for the given data sets.

This file can also be imported as a module and contains the following
functions:

    * colect_ct - function for calculating and collecting the cross-talk data into
    a .csv file.

    * plot_ct - function for plotting the cross-talk data from the .csv file as the
    cross-talk vs the distance between the two pixels, for which the cross-talk is
    calculated, in pixels.

"""

import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def collect_ct(path, pix: int, board_number: str, timestamps: int = 512):
    """Calculate cross-talk and save it into a .csv file.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : array-like, list
        Pixel numbers for which the cross-talk should be calculated.
    board_number : str
        The LinoSPAD2 daughterboard number.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    None.

    """
    print("\n> > > Collecting data for cross-talk analysis < < <\n")
    file_name_list = []
    pix1_list = []
    pix2_list = []
    timestamps_list = []
    deltas_list = []
    weight_list = []
    ct_list = []

    os.chdir(path)

    files = glob.glob("*.dat*")

    for i, file in enumerate(files):
        data = f_up.unpack_numpy(file, board_number, timestamps)
        pix1 = pix[0]
        for k, pix2 in enumerate(pix):
            if pix2 <= pix1:
                continue

            data_pair = np.vstack((data[pix1], data[pix2]))

            deltas = cd(data_pair, timestamps, -10e3, 10e3)

            timestamps_pix1 = len(np.where(data[pix1] > 0)[0])
            if timestamps_pix1 == 0:
                continue
            timestamps_pix2 = len(np.where(data[pix2] > 0)[0])

            weight = timestamps_pix1 / (timestamps_pix1 + timestamps_pix2)
            ct = len(deltas) * weight * 100 / timestamps_pix1

            file_name_list.append(file)
            pix1_list.append(pix1)
            pix2_list.append(pix2)
            timestamps_list.append(timestamps_pix1)
            deltas_list.append(len(deltas))
            weight_list.append(weight)
            ct_list.append(ct)

    dic = {
        "File": file_name_list,
        "Pixel 1": pix1_list,
        "Pixel 2": pix2_list,
        "Timestamps": timestamps_list,
        "Deltas": deltas_list,
        "Weight": weight_list,
        "CT": ct_list,
    }

    ct_data = pd.DataFrame(dic)

    path_to_save = "C:/Users/bruce/Documents/Quantum astrometry"
    os.chdir(path_to_save)

    if glob.glob("*CT_data.csv*") == []:
        ct_data.to_csv("CT_data.csv", index=False)
    else:
        ct_data.to_csv("CT_data.csv", mode="a", index=False, header=False)


def plot_ct(path, pix1, scale: str = "linear"):
    """Plot cross-talk data.

    Parameters
    ----------
    path : str
        Path to the folder where a .csv file with the cross-talk data is located.
    pix1 : int
        Pixel number relative to which the cross-talk data should be plotted.

    Returns
    -------
    None.

    """
    print("\n> > > Plotting cross-talk vs distance in pixels < < <\n")
    os.chdir(path)

    file = glob.glob("*CT_data.csv*")[0]

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

    xticks = np.arange(distance[0], distance[-1], 2)

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

    plt.savefig("{file}_{pix}".format(file=file[:-4], pix=pix1))
