import glob
import os
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd

path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585/SPDC"
path_FW2212 = path
os.chdir(path)
files = glob.glob("*.dat*")

# data = f_up.unpack_2212(files[0], board_number="A5", fw_ver="block", timestamps=80)

pix = (130, 131, 195, 196)

# data_pix = {}

# for i in range(len(pix)):
#     data_pix["{}".format(pix[i])] = np.array(data["{}".format(pix[i])])

# del data

# data_for_plot = {}

# same_y = False
# y_max_all = 0

# range_left = -15e3
# range_right = 15e3


# for q in range(len(pix)):
#     for w in range(len(pix)):
#         if w <= q:
#             continue
#         cycle = 0

#         data_for_plot["{p1}-{p2}".format(p1=pix[q], p2=pix[w])] = []

#         cyc2 = np.where(data_pix["{}".format(pix[w])] == -1)[0]
#         cyc2 = np.insert(cyc2, 0, -1)

#         for i, tmsp1 in enumerate(data_pix["{}".format(pix[q])]):
#             if tmsp1 == -1:
#                 cycle += 1
#                 continue
#             for j in range(cyc2[cycle] + 1, cyc2[cycle + 1]):
#                 tmsp2 = data_pix["{}".format(pix[w])][j]
#                 delta = np.abs(tmsp1 - tmsp2)
#                 if delta < 30e3:
#                     data_for_plot["{p1}-{p2}".format(p1=pix[q], p2=pix[w])].append(
#                         delta
#                     )


# plt.rcParams.update({"font.size": 22})
# if len(pix) > 2:
#     fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
# else:
#     fig = plt.figure(figsize=(14, 14))
# # check if the y limits of all plots should be the same
# if same_y is True:
#     y_max_all = 0

# for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
#     for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
#         if w <= q:
#             continue
#         if "Ne" and "540" in path:
#             chosen_color = "seagreen"
#         elif "Ne" and "656" in path:
#             chosen_color = "orangered"
#         elif "Ne" and "585" in path:
#             chosen_color = "goldenrod"
#         elif "Ar" in path:
#             chosen_color = "mediumslateblue"
#         else:
#             chosen_color = "salmon"

#         try:
#             bins = np.linspace(
#                 np.min(data_for_plot["{}-{}".format(pix[q], pix[w])]),
#                 np.max(data_for_plot["{}-{}".format(pix[q], pix[w])]),
#                 100,
#             )
#         except Exception:
#             print("Couldn't calculate bins: probably not enough delta ts.")
#             continue

#         if len(pix) > 2:
#             axs[q][w - 1].set_xlabel("\u0394t [ps]")
#             axs[q][w - 1].set_ylabel("Timestamps [-]")
#             n, b, p = axs[q][w - 1].hist(
#                 data_for_plot["{}-{}".format(pix[q], pix[w])],
#                 bins=bins,
#                 color=chosen_color,
#             )
#         else:
#             plt.xlabel("\u0394t [ps]")
#             plt.ylabel("Timestamps [-]")
#             n, b, p = plt.hist(
#                 data_for_plot["{}-{}".format(pix[q], pix[w])],
#                 bins=bins,
#                 color=chosen_color,
#             )

#         # find position of the histogram peak
#         try:
#             n_max = np.argmax(n)
#             arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
#         except Exception:
#             arg_max = None
#         if same_y is True:
#             try:
#                 y_max = np.max(n)
#             except ValueError:
#                 y_max = 0
#                 print("\nCould not find maximum y value\n")
#             if y_max_all < y_max:
#                 y_max_all = y_max
#             if len(pix) > 2:
#                 axs[q][w - 1].set_ylim(0, y_max + 4)
#             else:
#                 plt.ylim(0, y_max + 4)
#         if len(pix) > 2:
#             axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)

#             axs[q][w - 1].set_title(
#                 "Pixels {p1}-{p2}\nPeak position {pp}".format(
#                     p1=pix[q], p2=pix[w], pp=arg_max
#                 )
#             )
#         else:
#             plt.xlim(range_left - 100, range_right + 100)
#             plt.title("Pixels {p1}-{p2}".format(p1=pix[q], p2=pix[w]))
#         if same_y is True:
#             for q in range(len(pix)):
#                 for w in range(len(pix)):
#                     if w <= q:
#                         continue
#                     if len(pix) > 2:
#                         axs[q][w - 1].set_ylim(0, y_max_all + 10)
#                     else:
#                         plt.ylim(0, y_max_all + 10)

#         fig.tight_layout()

# =============================================================================
# Function
# =============================================================================


def plot_grid_mult_2212(
    path,
    pix,
    board_number: str,
    fw_ver: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = False,
):
    os.chdir(path)

    files = glob.glob("*.dat*")

    plot_name = files[0] + "-" + files[-1]

    data_for_plot = {}

    print("\n> > > Collecting data for the requested pixels < < <\n")

    for i, file in enumerate(files):
        data = f_up.unpack_2212(file, board_number, fw_ver, timestamps)

        data_pix = {}

        for i in range(len(pix)):
            data_pix["{}".format(pix[i])] = np.array(data["{}".format(pix[i])])

        del data

        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue
                # count cycles to move between them when calculating delta t
                cycle = 0

                data_for_plot["{p1}-{p2}".format(p1=pix[q], p2=pix[w])] = []

                # count cycles for the second pixel to use appropriate
                # timestamps when calculating delta t
                cyc2 = np.where(data_pix["{}".format(pix[w])] == -1)[0]
                cyc2 = np.insert(cyc2, 0, -1)

                for i, tmsp1 in enumerate(data_pix["{}".format(pix[q])]):
                    # '-1' indicate an end of a cycle
                    if tmsp1 == -1:
                        cycle += 1
                        continue
                    for j in range(cyc2[cycle] + 1, cyc2[cycle + 1]):
                        tmsp2 = data_pix["{}".format(pix[w])][j]
                        delta = tmsp1 - tmsp2
                        if delta < range_left:
                            continue
                        elif delta > range_right:
                            continue
                        else:
                            data_for_plot[
                                "{p1}-{p2}".format(p1=pix[q], p2=pix[w])
                            ].append(delta)

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

            fig.tight_layout()

            try:
                os.chdir("results/delta_t")
            except FileNotFoundError:
                os.makedirs("results/delta_t")
                os.chdir("results/delta_t")
            fig.tight_layout()  # for perfect spacing between the plots
            plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
            os.chdir("../..")


plot_grid_mult_2212(
    path_FW2212,
    pix=pix,
    board_number="A5",
    fw_ver="block",
    timestamps=50,
    range_left=-20e3,
    range_right=20e3,
)
