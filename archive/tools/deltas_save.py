import glob
import os
from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def deltas_save(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    os.chdir(path)

    files_all = glob.glob("*.dat*")

    plot_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    data_for_plot = {}

    # Collect the data for the required pixels
    print("\n> > > Collecting data for the requested pixels < < <\n")
    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        files_cut = files_all[i : (i + 1)]

        data_pix = f_up.unpack_mult_cut(files_cut, pix, board_number, timestamps)

        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue

                data_pair = np.vstack((data_pix[q], data_pix[w]))

                delta_ts = cd(
                    data_pair,
                    timestamps=timestamps,
                    range_left=range_left,
                    range_right=range_right,
                )
                if "{}-{}".format(pix[q], pix[w]) not in data_for_plot:
                    data_for_plot["{}-{}".format(pix[q], pix[w])] = list(delta_ts)
                else:
                    data_for_plot["{}-{}".format(pix[q], pix[w])].extend(delta_ts)

    data_for_plot_df = pd.DataFrame.from_dict(data_for_plot, orient="index")
    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        os.mkdir("delta_ts_data")
        os.chdir("delta_ts_data")
    data_for_plot_df.to_csv("{}_delta_t_data.csv".format(plot_name))
    os.chdir("..")


# =============================================================================
#
# =============================================================================

path = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585/for_tests"

deltas_save(
    path,
    pix=np.arange(135, 152, 1),
    board_number="A5",
    timestamps=80,
    range_left=-50e3,
    range_right=50e3,
)

from matplotlib import pyplot

os.chdir("D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585/for_tests/delta_ts_data")

data = pd.read_csv("0000010943-0000010952_delta_t_data.csv")
data = data.T

data1 = data[0]
data1 = data1.dropna()
data1 = data1.values[1:]

plt.hist(data1, bins=100)
plt.show()


# =============================================================================
#
# =============================================================================


def deltas_save_new(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    os.chdir(path)

    files_all = glob.glob("*.dat*")[:3]

    plot_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    deltas_all = {}
    for q in pix:
        for w in pix:
            if w <= q:
                continue
            deltas_all["{},{}".format(q, w)] = []

    # Collect the data for the required pixels
    print("\n> > > Collecting data for the requested pixels < < <\n")
    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        data = f_up.unpack_dict(file, board_number="A5", timestamps=timestamps, pix=pix)

        for q in data.keys():
            for w in data.keys():
                if w <= q:
                    continue

                if len(data[q]) >= len(data[w]):
                    cycle = 0
                    cyc2 = np.argwhere(data[w] < 0)
                    cyc2 = np.insert(cyc2, 0, -1)
                    for i, tmsp1 in enumerate(data[q]):
                        if tmsp1 == -1:
                            cycle += 1
                            continue
                        deltas = data[w][cyc2[cycle] + 1 : cyc2[cycle + 1]] - tmsp1
                        ind = np.where(np.abs(deltas) < 50e3)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])
                else:
                    cycle = 0
                    cyc2 = np.argwhere(data[q] < 0)
                    cyc2 = np.insert(cyc2, 0, -1)
                    for i, tmsp1 in enumerate(data[w]):
                        if tmsp1 == -1:
                            cycle += 1
                            continue
                        deltas = data[q][cyc2[cycle] + 1 : cyc2[cycle + 1]] - tmsp1
                        ind = np.where(np.abs(deltas) < 50e3)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])

    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        os.mkdir("delta_ts_data")
        os.chdir("delta_ts_data")
    data_for_plot_df.to_csv(
        "{}_delta_t_data_new.csv".format(plot_name), index=False, index_label=False
    )
    os.chdir("..")


deltas_save_new(path, pix=np.arange(134, 152, 1), board_number="A5", timestamps=80)


os.chdir("D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585/for_tests/delta_ts_data")

data2 = pd.read_csv("0000010943-0000010952_delta_t_data_new.csv", index_col=0).T

data22 = data2["135,136"]
data22 = data22.dropna()
data22 = data22.values[1:]

plt.figure()
plt.hist(data22, bins=100)
plt.show()
