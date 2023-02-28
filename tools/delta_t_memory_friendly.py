import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from functions.calc_diff import calc_diff as cd
from functions import unpack as f_up
from tqdm import tqdm
from math import ceil

path = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585"

pix = (135, 136, 189, 190)
timestamps = 80
board_number = "A5"
range_left = -20e3
range_right = 20e3
same_y = False

os.chdir(path)

files_all = glob("*.dat*")

plot_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

data_for_plot = {}

for i in range(ceil(len(files_all) / 10)):
    files_cut = files_all[i * 10 : (i + 1) * 10]

    data_pix = f_up.unpack_mult_cut(files_cut, pix, board_number, timestamps)

    for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
        for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
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

plt.rcParams.update({"font.size": 22})
if len(pix) > 2:
    fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
else:
    fig = plt.figure(figsize=(14, 14))
# check if the y limits of all plots should be the same
if same_y is True:
    y_max_all = 0

for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
    for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
        if w <= q:
            continue
        if "Ne" and "540" in path:
            chosen_color = "seagreen"
        elif "Ne" and "656" in path:
            chosen_color = "orangered"
        elif "Ne" and "585" in path:
            chosen_color = "goldenrod"
        elif "Ar" in path:
            chosen_color = "mediumslateblue"
        else:
            chosen_color = "salmon"
        try:
            bins = np.linspace(
                np.min(data_for_plot["{}-{}".format(pix[q], pix[w])]),
                np.max(data_for_plot["{}-{}".format(pix[q], pix[w])]),
                100,
            )
        except Exception:
            print("Couldn't calculate bins: probably not enough delta ts.")
            continue
        if len(pix) > 2:
            axs[q][w - 1].set_xlabel("\u0394t [ps]")
            axs[q][w - 1].set_ylabel("Timestamps [-]")
            n, b, p = axs[q][w - 1].hist(
                data_for_plot["{}-{}".format(pix[q], pix[w])],
                bins=bins,
                color=chosen_color,
            )
        else:
            plt.xlabel("\u0394t [ps]")
            plt.ylabel("Timestamps [-]")
            n, b, p = plt.hist(
                data_for_plot["{}-{}".format(pix[q], pix[w])],
                bins=bins,
                color=chosen_color,
            )
        # find position of the histogram peak
        try:
            n_max = np.argmax(n)
            arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
        except Exception:
            arg_max = None
        if same_y is True:
            try:
                y_max = np.max(n)
            except ValueError:
                y_max = 0
                print("\nCould not find maximum y value\n")
            if y_max_all < y_max:
                y_max_all = y_max
            if len(pix) > 2:
                axs[q][w - 1].set_ylim(0, y_max + 4)
            else:
                plt.ylim(0, y_max + 4)
        if len(pix) > 2:
            axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)

            axs[q][w - 1].set_title(
                "Pixels {p1}-{p2}\nPeak position {pp}".format(
                    p1=pix[q], p2=pix[w], pp=arg_max
                )
            )
        else:
            plt.xlim(range_left - 100, range_right + 100)
            plt.title("Pixels {p1}-{p2}".format(p1=pix[q], p2=pix[w]))
        if same_y is True:
            for q in range(len(pix)):
                for w in range(len(pix)):
                    if w <= q:
                        continue
                    if len(pix) > 2:
                        axs[q][w - 1].set_ylim(0, y_max_all + 10)
                    else:
                        plt.ylim(0, y_max_all + 10)
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.makedirs("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
        os.chdir("../..")
