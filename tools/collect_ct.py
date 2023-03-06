"""

"""

import glob
import os

import numpy as np
import pandas as pd

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def collect_ct(path, pix, board_number: str, timestamps: int = 512):
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
