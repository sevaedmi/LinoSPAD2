import glob
import os
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def plot_grid_mult(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = False,
):
    """
    Plot a grid of delta ts for all pairs of given pixels.

    Function for calculating timestamp differences and plotting them in a grid. Works
    with multiple .dat files, keeping the data only for the required pixels and thus
    reducing the memory occupied.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : array-like
        Pixel numbers for which the timestamps differences should be calculated.
    board_number : str
        The LinoSPAD2 board number. Input required for using the calibration data.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.
    range_left : float, optional
        Left border of the window in which the timestamp differences should be
        calculated. The default is -2.5e3.
    range_right : float, optional
        Right border of the windot in which the timestamp differences should be
        calculated. The default is 2.5e3.
    show_fig : bool, optional
        Switch for showing the plots. The default is False.
    same_y : bool, optional
        Switch for equalizing the y axis for all plots. The default is False.

    Returns
    -------
    None.

    """
    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    files_all = glob.glob("*.dat*")

    plot_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    data_for_plot = {}

    pix = np.array(pix)
    pix = np.insert(pix, 0, pix[0] - 1)
    pix = np.insert(pix, 2, pix[2] - 1)

    # Collect the data for the required pixels
    print("\n> > > Collecting data for the requested pixels < < <\n")
    for i in tqdm(range(ceil(len(files_all) / 5)), desc="Collecting data"):
        files_cut = files_all[i * 5 : (i + 1) * 5]

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

    plt.rcParams.update({"font.size": 22})

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

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
        bins1 = np.linspace(
            np.min(data_for_plot["{}-{}".format(pix[1], pix[0])]),
            np.max(data_for_plot["{}-{}".format(pix[1], pix[0])]),
            100,
        )
        # TODO: cut
        print(bins1)
    except Exception:
        print("Couldn't calculate bins1: probably not enough delta ts.")
        pass

    try:
        bins2 = np.linspace(
            np.min(data_for_plot["{}-{}".format(pix[-1], pix[-2])]),
            np.max(data_for_plot["{}-{}".format(pix[-1], pix[-2])]),
            100,
        )
        # TODO: cut
        print(bins2)
    except Exception:
        print("Couldn't calculate bins2: probably not enough delta ts.")
        pass

    try:
        bins3 = np.linspace(
            np.min(data_for_plot["{}-{}".format(pix[1], pix[-1])]),
            np.max(data_for_plot["{}-{}".format(pix[1], pix[-1])]),
            100,
        )
        # TODO: cut
        print(bins3)
    except Exception:
        print("Couldn't calculate bins3: probably not enough delta ts.")
        pass

    axs[0].set_xlabel("\u0394t [ps]")
    axs[0].set_ylabel("Timestamps [-]")
    n, b, p = axs[0].hist(
        data_for_plot["{}-{}".format(pix[1], pix[0])],
        bins=bins1,
        color=chosen_color,
    )
    try:
        peak_max_pos = np.argmax(n).astype(np.intc)
        # 2 ns window around peak
        win = int(1000 / ((range_right - range_left) / 100))
        peak_max_1 = np.sum(n[peak_max_pos - win : peak_max_pos + win])
    except Exception:
        peak_max_1 = None
    axs[0].set_xlim(range_left - 100, range_right + 100)
    axs[0].set_title(
        "Pixels {p1},{p2}\nPeak in 2 ns window: {pm}".format(
            p1=pix[1], p2=pix[0], pm=int(peak_max_1)
        )
    )

    axs[1].set_xlabel("\u0394t [ps]")
    axs[1].set_ylabel("Timestamps [-]")
    n, b, p = axs[1].hist(
        data_for_plot["{}-{}".format(pix[-1], pix[-2])],
        bins=bins2,
        color=chosen_color,
    )
    try:
        peak_max_pos = np.argmax(n).astype(np.intc)
        # 2 ns window around peak
        win = int(1000 / ((range_right - range_left) / 100))
        peak_max_1 = np.sum(n[peak_max_pos - win : peak_max_pos + win])
    except Exception:
        peak_max_1 = None
    axs[1].set_xlim(range_left - 100, range_right + 100)
    axs[1].set_title(
        "Pixels {p1},{p2}\nPeak in 2 ns window: {pm}".format(
            p1=pix[-1], p2=pix[-2], pm=int(peak_max_1)
        )
    )

    axs[2].set_xlabel("\u0394t [ps]")
    axs[2].set_ylabel("Timestamps [-]")
    n, b, p = axs[2].hist(
        data_for_plot["{}-{}".format(pix[1], pix[-1])],
        bins=bins3,
        color=chosen_color,
    )
    try:
        peak_max_pos = np.argmax(n).astype(np.intc)
        # 2 ns window around peak
        win = int(1000 / ((range_right - range_left) / 100))
        peak_max_1 = np.sum(n[peak_max_pos - win : peak_max_pos + win])
    except Exception:
        peak_max_1 = None
    axs[3].set_xlim(range_left - 100, range_right + 100)
    axs[3].set_title(
        "Pixels {p1},{p2}\nPeak in 2 ns window: {pm}".format(
            p1=pix[1], p2=pix[-1], pm=int(peak_max_1)
        )
    )

    # if same_y is True:
    #     try:
    #         y_max = np.max(n)
    #     except ValueError:
    #         y_max = 0
    #         print("\nCould not find maximum y value\n")
    #     if y_max_all < y_max:
    #         y_max_all = y_max
    #     if len(pix) > 2:
    #         axs[q][w - 1].set_ylim(0, y_max + 4)
    #     else:
    #         plt.ylim(0, y_max + 4)

    # if same_y is True:
    #     for q in range(len(pix)):
    #         for w in range(len(pix)):
    #             if w <= q:
    #                 continue
    #             if len(pix) > 2:
    #                 axs[q][w - 1].set_ylim(0, y_max_all + 10)
    #             else:
    #                 plt.ylim(0, y_max_all + 10)
    try:
        os.chdir("results/delta_t")
    except FileNotFoundError:
        os.makedirs("results/delta_t")
        os.chdir("results/delta_t")
    fig.tight_layout()  # for perfect spacing between the plots
    plt.savefig("{name}_delta_t.png".format(name=plot_name))
    os.chdir("../..")


path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585"
plot_grid_mult(path, pix=(138, 156), board_number="A5", timestamps=100)
