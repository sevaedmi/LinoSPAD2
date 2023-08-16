"""Module for calculating and plotting the timestamp differences.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 binary data output.

This file can also be imported as a module and contains the following
functions:

    * deltas_save_numpy - unpacks the binary data, calculates timestamp
    differences and saves into a .csv file. Works with firmware versions
    '2208' and '2212b'.

    * delta_cp - collect timestamps from the .csv file and plot them in
    a grid.
"""
import glob
import os
import sys
import time
from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up


def deltas_save_numpy(
    path,
    pix,
    rewrite: bool,
    board_number: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
):
    """Calculate and save timestamp differences into .csv file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels and saves them into a .csv table. Works with
    firmware versions '2208' and '2212b' (block). The plot is saved
    in the 'results' folder, which is created (if it does not already
    exist) in the same folder where data are.tel

    Parameters
    ----------
    path : str
        Path to data files.
    pix : list
        List of pixel numbers for which the timestamp differences should
        be calculate and saved.
    rewrite : bool
        Switch for rewriting the csv file if it already exists.
    board_number : str
        The LinoSPAD2 daughterboard number.
    fw_ver: str
        LinoSPAD2 firmware version. Versions "2208" and "2212b (block)"
        are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of 'fw_ver'
        are accepted. First error is raised so that the plot does not
        accidentally gets rewritten in the case no clear input was
        given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212b' or '2208'")
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(board_number, str) is not True:
        raise TypeError(
            "'board_number' should be string, either 'NL11' or 'A5'"
        )
    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file already exists and will be"
                    "rewritten. ! ! !\n"
                )
                for i in range(5):
                    print(
                        "\n! ! ! Deleting the file in {} ! ! !\n".format(5 - i)
                    )
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                sys.exit(
                    "\n csv file already exists, 'rewrite' set to"
                    "'False', exiting."
                )
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    print("\n> > > Collecting data for the requested pixels < < <\n")
    if fw_ver == "2208":
        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}
            for q in pix:
                for w in pix:
                    if w <= q:
                        continue
                    deltas_all["{},{}".format(q, w)] = []

            # Unpack data for the requested pixels into dictionary
            data = f_up.unpack_numpy_dict(
                file, board_number=board_number, timestamps=timestamps, pix=pix
            )

            # Calculate and collect timestamp differences
            for q in pix:
                for w in pix:
                    if w <= q:
                        continue

                    # Follows the cycle number in the first array
                    cycle = 0
                    # Follows the cycle number in the second array
                    cyc2 = np.argwhere(data["{}".format(w)] < 0)
                    cyc2 = np.insert(cyc2, 0, -1)
                    for i, tmsp1 in enumerate(data["{}".format(q)]):
                        if tmsp1 == -2:
                            cycle += 1
                            continue
                        deltas = (
                            data["{}".format(w)][
                                cyc2[cycle] + 1 : cyc2[cycle + 1]
                            ]
                            - tmsp1
                        )
                        # Collect deltas in the requested window
                        ind = np.where(np.abs(deltas) < delta_window)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])

            # Save data as a .csv file
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")
            csv_file = glob.glob("*{}*".format(out_file_name))
            # create for first file, append for all next ones
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name), mode="a", header=False
                )
            else:
                data_for_plot_df.to_csv("{}.csv".format(out_file_name))
            os.chdir("..")
    elif fw_ver == "2212b":
        # for transforming pixel number into TDC number + pixel
        # coordinates in that TDC
        pix_coor = np.arange(256).reshape(64, 4)

        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}

            # Unpack data for the requested pixels into dictionary
            data_all = f_up.unpack_2212_numpy(file, board_number, timestamps)

            # Calculate and collect timestamp differences
            for q in pix:
                for w in pix:
                    if w <= q:
                        continue
                    deltas_all["{},{}".format(q, w)] = []
                    # find end of cycles
                    cycler = np.argwhere(data_all[0].T[0] == -2)
                    cycler = np.insert(cycler, 0, 0)
                    # first pixel in the pair
                    tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
                    pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                    # second pixel in the pair
                    tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                    pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                    # get timestamp for both pixels in the given cycle
                    for cyc in range(len(cycler) - 1):
                        pix1_ = pix1[
                            np.logical_and(
                                pix1 > cycler[cyc], pix1 < cycler[cyc + 1]
                            )
                        ]
                        if not np.any(pix1_):
                            continue
                        pix2_ = pix2[
                            np.logical_and(
                                pix2 > cycler[cyc], pix2 < cycler[cyc + 1]
                            )
                        ]
                        if not np.any(pix2_):
                            continue
                        # calculate delta t
                        tmsp1 = data_all[tdc1].T[1][
                            pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                        ]
                        tmsp2 = data_all[tdc2].T[1][
                            pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                        ]
                        for t1 in tmsp1:
                            deltas = tmsp2 - t1
                            ind = np.where(np.abs(deltas) < delta_window)[0]
                            deltas_all["{},{}".format(q, w)].extend(
                                deltas[ind]
                            )

            # Save data as a .csv file
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")
            csv_file = glob.glob("*{}.csv*".format(out_file_name))
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name),
                    mode="a",
                    index=False,
                    header=False,
                )
            else:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name), index=False
                )
            os.chdir("..")


def delta_cp(
    path,
    pix,
    rewrite: bool,
    range_left: int = -10e3,
    range_right: int = 10e3,
    same_y: bool = False,
):
    """Plot timestamp differences from .csv file.

    Plots timestamp differences from a .csv file as a grid of histograms
    and as a single plot.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : list
        List of pixel numbers for which the timestamp differences
        should be plotted.
    rewrite : bool
        Switch for rewriting the plot if it already exists.
    range_left : int, optional
        Lower limit for timestamp differences, lower values are not used.
        The default is -10e3.
    range_right : int, optional
        Upper limit for timestamp differences, higher values are no used.
        The default is 10e3.
    same_y : bool, optional
        Switch for plotting the histograms with the same y axis.
        The default is False.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' are accepted. The error is
        raised so that the plot does not accidentally gets rewritten.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    plt.ioff()
    os.chdir(path)

    files_all = glob.glob("*.dat*")
    csv_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if plot exists and if it should be rewrited
    try:
        os.chdir("results/delta_t")
        if os.path.isfile(
            "{name}_delta_t_grid.png".format(name=csv_file_name)
        ):
            if rewrite is True:
                print(
                    "\n! ! ! Plot already exists and will be"
                    "rewritten. ! ! !\n"
                )
            else:
                sys.exit(
                    "\nPlot already exists, 'rewrite' set to 'False', exiting."
                )
        os.chdir("../..")
    except FileNotFoundError:
        pass

    plt.rcParams.update({"font.size": 22})

    if len(pix) > 2:
        fig, axs = plt.subplots(
            len(pix) - 1,
            len(pix) - 1,
            figsize=(5.5 * len(pix), 5.5 * len(pix)),
        )
        for ax in axs:
            for x in ax:
                x.axes.set_axis_off()
    else:
        fig = plt.figure(figsize=(14, 14))

    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    for q in tqdm(range(len(pix)), desc="Row in plot"):
        for w in range(len(pix)):
            if w <= q:
                continue
            if len(pix) > 2:
                axs[q][w - 1].axes.set_axis_on()
            try:
                # keep only the required column in memory
                data_to_plot = pd.read_csv(
                    "delta_ts_data/{}.csv".format(csv_file_name),
                    usecols=["{},{}".format(pix[q], pix[w])],
                ).dropna()
            except ValueError:
                continue

            # prepare the data for plot
            data_to_plot = np.array(data_to_plot)
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot < range_left)
            )
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot > range_right)
            )

            try:
                bins = np.linspace(
                    np.min(data_to_plot),
                    np.max(data_to_plot),
                    100,
                )
            except ValueError:
                print(
                    "\nCouldn't calculate bins: probably not enough delta ts."
                )
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

            if len(pix) > 2:
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("# of coincidences [-]")
                n, b, p = axs[q][w - 1].hist(
                    data_to_plot,
                    bins=bins,
                    color=chosen_color,
                )
            else:
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("# of coincidences [-]")
                n, b, p = plt.hist(
                    data_to_plot,
                    bins=bins,
                    color=chosen_color,
                )

            try:
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                peak_max = np.sum(n[peak_max_pos - win : peak_max_pos + win])
            except ValueError:
                peak_max = None

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
                    "Pixels {p1},{p2}\nPeak in 2 ns window: {pp}".format(
                        p1=pix[q], p2=pix[w], pp=int(peak_max)
                    )
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)
                plt.title("Pixels {p1},{p2}".format(p1=pix[q], p2=pix[w]))

            try:
                os.chdir("results/delta_t")
            except FileNotFoundError:
                os.makedirs("results/delta_t")
                os.chdir("results/delta_t")
            fig.tight_layout()  # for perfect spacing between the plots
            plt.savefig("{name}_delta_t_grid.png".format(name=csv_file_name))
            os.chdir("../..")
