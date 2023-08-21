"""Module that contains functions cut from the 'functions' as these
are no longer utilized, only for debugging.

Following functions can be found in this module.

    * collect_ct
"""

import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    output is saved as a .csv file in the folder "/cross_talk_data",
    which is created if it does not exist, in the same folder where
    datafiles are located.

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
        raise TypeError(
            "'board_number' should be string, either 'NL11' or 'A5'"
        )
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
        for i, file in enumerate(tqdm(files)):
            data = f_up.unpack_2208_numpy(file, board_number, timestamps)
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
        for i, file in enumerate(tqdm(files)):
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

                deltas = cd.calc_diff_2212(
                    data1, data2, cycle_ends, delta_window
                )

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

    print(
        "\n> > > Saving data as 'CT_data_{}-{}.csv' in"
        " {path} < < <\n".format(
            files[0], files[-1], path=path + "/cross_talk_data"
        )
    )

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

    try:
        os.chdir(path + "/cross_talk_data")
    except FileNotFoundError:
        os.makedirs("{}".format(path + "/cross_talk_data"))
        os.chdir(path + "/cross_talk_data")

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
