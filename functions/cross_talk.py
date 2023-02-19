""" A set of functions to calculate and collect the cross-talk data
for the given data sets.

    * colect_ct - function for calculating and collecting the cross-talk data into
    a .csv file.
    * plot_ct - function for plotting the cross-talk data from the .csv file as the
    cross-talk vs the distance between the two pixels, for which the cross-talk is
    calculated, in pixels.

"""

import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def collect_ct(path, pix: int, board_number: str, timestamps: int = 512):

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

        data = f_up.unpack_calib(file, board_number, timestamps)
        pix1 = pix
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


def plot_ct(path, pix1):

    path = "C:/Users/bruce/Documents/Quantum astrometry"

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

    plt.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    ax1.errorbar(distance, ct, yerr=yerr, color="salmon")
    ax1.set_xlabel("Distance in pixels [-]")
    ax1.set_ylabel("Average cross-talk [%]")
    ax1.set_title("Pixel {}".format(pix1))

    plt.savefig("{file}_{pix}".format(file=file[:-4], pix=pix1))
