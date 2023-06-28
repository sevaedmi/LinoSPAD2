"""

"""

import glob
import os

import numpy as np
import pandas as pd

from functions import calc_diff as cd
from functions import unpack as f_up


def collect_ct_2208(path, pix, board_number: str, timestamps: int = 512):
    file_name_list = []
    pix1_list = []
    pix2_list = []
    timestamps_list = []
    deltas_list = []
    ct_list = []

    os.chdir(path)

    files = glob.glob("*.dat*")

    for i, file in enumerate(files):
        data = f_up.unpack_calib(file, board_number, timestamps)
        # for j, pix1 in enumerate(pix):
        pix1 = 135
        for k, pix2 in enumerate(pix):
            if pix2 <= pix1:
                continue

            data_pair = np.vstack((data[pix1], data[pix2]))

            deltas = cd(data_pair, timestamps, -15e3, 15e3)

            timestamps_pix1 = len(np.where(data[pix1] > 0)[0])
            if timestamps_pix1 == 0:
                continue
            timestamps_pix2 = len(np.where(data[pix2] > 0)[0])

            ct = len(deltas) * 100 / (timestamps_pix1 + timestamps_pix2)

            file_name_list.append(file)
            pix1_list.append(pix1)
            pix2_list.append(pix2)
            timestamps_list.append(timestamps_pix1)
            deltas_list.append(len(deltas))
            ct_list.append(ct)

    dic = {
        "File": file_name_list,
        "Pixel 1": pix1_list,
        "Pixel 2": pix2_list,
        "Timestamps": timestamps_list,
        "Deltas": deltas_list,
        "CT": ct_list,
    }

    ct_data = pd.DataFrame(dic)

    path_to_save = "C:/Users/bruce/Documents/Quantum astrometry"
    os.chdir(path_to_save)

    if glob.glob("*CT_data.csv*") == []:
        ct_data.to_csv("CT_data.csv", index=False)
    else:
        ct_data.to_csv("CT_data.csv", mode="a", index=False, header=False)


def collect_ct_2212(
    path,
    pix,
    pixels: list,
    board_number: str,
    timestamps: int = 512,
    delta_window: float = 15e3,
):
    """


    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    pix : TYPE
        DESCRIPTION.
    pixels : list
        DESCRIPTION.
    board_number : str
        DESCRIPTION.
    timestamps : int, optional
        DESCRIPTION. The default is 512.
    delta_window : float, optional
        DESCRIPTION. The default is 15e3.

    Returns
    -------
    None.

    """
    file_name_list = []
    timestamps_list1 = []
    timestamps_list2 = []
    pix1_list = []
    pix2_list = []
    deltas_list = []
    ct_list = []

    os.chdir(path)

    files = glob.glob("*.dat*")

    pix_coor = np.arange(256).reshape(64, 4)

    for i, file in enumerate(files):
        data_all = f_up.unpack_2212_numpy(file, board_number, timestamps)

        tdc1, pix_c1 = np.argwhere(pix_coor == pix)[0]
        pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
        data1 = data_all[tdc1].T[1][pix1]

        cycle_ends = np.argwhere(data_all[0].T[0] == -2)
        cycle_ends = np.insert(cycle_ends, 0, 0)
        for j, pix_ in enumerate(pixels):
            if pix_ <= pix:
                continue

            tdc2, pix_c2 = np.argwhere(pix_coor == pix_)[0]
            pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]

            data2 = data_all[tdc2].T[1][pix2]

            deltas = cd.calc_diff_2212(data1, data2, cycle_ends, delta_window)

            timestamps_pix1 = len(np.where(data1 > 0)[0])
            if timestamps_pix1 == 0:
                continue
            timestamps_pix2 = len(np.where(data2 > 0)[0])

            ct = len(deltas) * 100 / (timestamps_pix1 + timestamps_pix2)

            file_name_list.append(file)
            pix1_list.append(pix)
            pix2_list.append(pix_)
            timestamps_list1.append(timestamps_pix1)
            timestamps_list2.append(timestamps_pix2)
            deltas_list.append(len(deltas))
            ct_list.append(ct)

    dic = {
        "File": file_name_list,
        "Pixel 1": pix1_list,
        "Pixel 2": pix2_list,
        "Timestamps1": timestamps_list1,
        "Timestamps2": timestamps_list2,
        "Deltas": deltas_list,
        "CT": ct_list,
    }

    ct_data = pd.DataFrame(dic)

    path_to_save = "C:/Users/bruce/Documents/Quantum astrometry"
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
