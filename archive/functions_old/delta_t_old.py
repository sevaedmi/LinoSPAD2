"""Module that contains functions cut from the 'functions' as these
are no longer utilized, only for debugging.

Following functions can be found in this module.

    * plot_grid_mult
    * plot_grid_mult_2212
    * deltas_save
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

from functions import calc_diff as cd
from functions import unpack as f_up


def plot_grid_mult(
    path,
    pix,
    board_number: str,
    rewrite: bool,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = False,
):
    """Plot a grid of delta ts for all pairs of given pixels.

    Function for calculating timestamp differences and plotting them in
    a grid. Works with multiple .dat files, keeping the data only for
    the required pixels and thus reducing the memory occupied. Works
    with fimware version 2208.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : array-like
        Pixel numbers for which the timestamps differences should be
        calculated.
    board_number : str
        The LinoSPAD2 board number. Input required for using the
        calibration data.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The
        default is 512.
    range_left : float, optional
        Left border of the window in which the timestamp differences
        should be calculated. The default is -2.5e3.
    range_right : float, optional
        Right border of the windot in which the timestamp differences
        should be calculated. The default is 2.5e3.
    show_fig : bool, optional
        Switch for showing the plots. The default is False.
    same_y : bool, optional
        Switch for equalizing the y axis for all plots. The default is
        False.

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
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")

    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    files_all = glob.glob("*.dat*")

    plot_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if plot exists and if it should be rewrited
    try:
        os.chdir("results/delta_t")
        if os.path.isfile("{name}_delta_t_grid.png".format(name=plot_name)):
            if rewrite is True:
                print("\n> > > Plot already exists and will be" "rewritten. < < <\n")
            else:
                print("\n> > > Plot already exists. < < <\n")
                sys.exit("Plot already exists, 'rewrite' set to 'False'.")
    except FileNotFoundError:
        pass
    os.chdir("../..")

    # prepare dictionary for delta ts to plot
    data_for_plot = {}

    # Collect the data for the required pixels
    print("\n> > > Collecting data for the requested pixels < < <\n")
    for i in tqdm(range(len(files_all)), desc="Collecting data"):
        file = files_all[i]

        data_pix = f_up.unpack_numpy(file, board_number, timestamps, pix)

        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue

                data_pair = np.vstack((data_pix[q], data_pix[w]))

                delta_ts = cd.calc_diff_2208(
                    data_pair,
                    timestamps=timestamps,
                    range_left=range_left,
                    range_right=range_right,
                )
                if "{},{}".format(pix[q], pix[w]) not in data_for_plot:
                    data_for_plot["{},{}".format(pix[q], pix[w])] = list(delta_ts)
                else:
                    data_for_plot["{},{}".format(pix[q], pix[w])].extend(delta_ts)

    plt.rcParams.update({"font.size": 22})
    # figure manipulations based on number of plots in the resulting grid
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

    # Calculate delta ts and plot them
    print("\n> > > Calculating the timestamp differences < < <\n")
    for q in tqdm(range(len(pix) - 1), desc="Minuend pixel   "):
        for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
            if w <= q:
                continue
            axs[q][w - 1].axes.set_axis_on()
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
                    np.min(data_for_plot["{},{}".format(pix[q], pix[w])]),
                    np.max(data_for_plot["{},{}".format(pix[q], pix[w])]),
                    100,
                )
            except Exception:
                print("Couldn't calculate bins: probably not enough delta ts.")
                continue
            if len(pix) > 2:
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("Timestamps [-]")
                n, b, p = axs[q][w - 1].hist(
                    data_for_plot["{},{}".format(pix[q], pix[w])],
                    bins=bins,
                    color=chosen_color,
                )
            else:
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n, b, p = plt.hist(
                    data_for_plot["{},{}".format(pix[q], pix[w])],
                    bins=bins,
                    color=chosen_color,
                )
            try:
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                peak_max = np.sum(n[peak_max_pos - win : peak_max_pos + win])
            except Exception:
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
                    "Pixels {p1},{p2}\nPeak in 2 ns window: {pm}".format(
                        p1=pix[q], p2=pix[w], pm=int(peak_max)
                    )
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)
                plt.title("Pixels {p1},{p2}".format(p1=pix[q], p2=pix[w]))
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
            plt.savefig("{name}_delta_t_grid.png".format(name=plot_name))
            os.chdir("../..")


def plot_grid_mult_2212(
    path,
    pix,
    board_number: str,
    fw_ver: str,
    rewrite: bool,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = False,
):
    """Plot a grid of delta ts for all pairs of given pixels.

    Plot a grid of timestamp differences for all pairs in the given
    pixel range. Works with firmware version '2212'. The plot is saved
    in the 'results' folder, which is created (if it does not already
    exist) in the same folder where data are.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : array-like
        Array of pixel numbers for which the timestamp differences
        should be calculated and plotted.
    board_number : str
        The LinoSPAD2 daughterboard number.
    fw_ver : str
        The version of the 2212 firmware: '2212b' or '2212s'.
    rewrite : bool
        Switch for rewriting the plot if it already exists.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The
        default is 512.
    range_left : float, optional
        Lower limit for timestamp differences. The default is -2.5e3.
    range_right : float, optional
        Upper limit for timestamp differences. The default is 2.5e3.
    show_fig : bool, optional
        Switch for showing figure after plotting. The default is False.
    same_y : bool, optional
        Switch for making the y axis the same for all plots. The default
        is True.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' are accepted. The error is
        raised so that the plot does not accidentally gets rewritten in
        the case no clear input was given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")

    os.chdir(path)

    files = glob.glob("*.dat*")

    plot_name = files[0] + "-" + files[-1]

    data_for_plot = {}

    print("\n> > > Collecting data for the requested pixels < < <\n")

    for q in range(len(pix)):
        for w in range(len(pix)):
            if w <= q:
                continue
            data_for_plot["{p1},{p2}".format(p1=pix[q], p2=pix[w])] = []

    for i in tqdm(range(len(files)), desc="Collecting delta ts"):
        file = files[i]
        data = f_up.unpack_2212(file, board_number, fw_ver, timestamps)

        data_pix = {}

        for j in range(len(pix)):
            data_pix["{}".format(pix[j])] = np.array(data["{}".format(pix[j])])

        del data

        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue
                # count cycles to move between them when calculating delta t
                cycle = 0

                # count cycles for the second pixel to use appropriate
                # timestamps when calculating delta t
                cyc2 = np.where(data_pix["{}".format(pix[w])] == -1)[0]
                cyc2 = np.insert(cyc2, 0, -1)

                for k, tmsp1 in enumerate(data_pix["{}".format(pix[q])]):
                    # '-1' indicate an end of a cycle
                    if tmsp1 == -2:
                        cycle += 1
                        continue
                    for n in range(cyc2[cycle] + 1, cyc2[cycle + 1]):
                        tmsp2 = data_pix["{}".format(pix[w])][n]
                        delta = tmsp1 - tmsp2
                        if delta < range_left:
                            continue
                        elif delta > range_right:
                            continue
                        else:
                            data_for_plot[
                                "{p1},{p2}".format(p1=pix[q], p2=pix[w])
                            ].append(delta)

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

    for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
        for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
            if w <= q:
                continue
            axs[q][w - 1].axes.set_axis_on()
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
                    np.min(data_for_plot["{},{}".format(pix[q], pix[w])]),
                    np.max(data_for_plot["{},{}".format(pix[q], pix[w])]),
                    100,
                )
            except Exception:
                print("Couldn't calculate bins: probably not enough delta ts.")
                continue

            if len(pix) > 2:
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("Timestamps [-]")
                n, b, p = axs[q][w - 1].hist(
                    data_for_plot["{},{}".format(pix[q], pix[w])],
                    bins=bins,
                    color=chosen_color,
                )
            else:
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n, b, p = plt.hist(
                    data_for_plot["{},{}".format(pix[q], pix[w])],
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
                    "Pixels {p1},{p2}\nPeak position {pp}".format(
                        p1=pix[q], p2=pix[w], pp=arg_max
                    )
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)
                plt.title("Pixels {p1},{p2}".format(p1=pix[q], p2=pix[w]))

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
            plt.savefig("{name}_delta_t_grid.png".format(name=plot_name))
            os.chdir("../..")


def deltas_save(
    path,
    pix,
    rewrite: bool,
    board_number: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
):
    """Calculate and save timestamp differences into .csv file.

    Unpacks data into a dictionary, calculated timestamp differences for
    the requested pixels and saves them into a .csv table.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved.
    rewrite : bool
        Switch for rewriting the csv file if it already exists.
    board_number : str
        The LinoSPAD2 daughterboard number.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window against which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).

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
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")
    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file already exists and will be" "rewritten. ! ! !\n"
                )
                for i in range(5):
                    print("\n! ! ! Deleting the file in {} ! ! !\n".format(i))
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                # print("\n> > > Plot already exists. < < <\n")
                sys.exit(
                    "\n csv file already exists, 'rewrite' set to" "'False', exiting."
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
                            data["{}".format(w)][cyc2[cycle] + 1 : cyc2[cycle + 1]]
                            - tmsp1
                        )
                        # Collect deltas in the requested window
                        ind = np.where(np.abs(deltas) < delta_window)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])
            deltas_all = {
                k: v
                for k, v in deltas_all.items()
                if len(deltas_all[k]) > 10 and v != []
            }

            # Save data as a .csv file
            data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")
            csv_file = glob.glob("*{}*".format(out_file_name))
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name), mode="a", index=False
                )
            else:
                data_for_plot_df.to_csv("{}.csv".format(out_file_name))
            os.chdir("..")
    elif fw_ver == "2212b":
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
            data = f_up.unpack_2212(
                file,
                board_number=board_number,
                fw_ver="2212b",
                timestamps=timestamps,
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
                            data["{}".format(w)][cyc2[cycle] + 1 : cyc2[cycle + 1]]
                            - tmsp1
                        )
                        # Collect deltas in the requested window
                        ind = np.where(np.abs(deltas) < delta_window)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])

            deltas_all = {
                k: v
                for k, v in deltas_all.items()
                if len(deltas_all[k]) > 10 and v != []
            }

            # Save data as a .csv file
            data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")
            csv_file = glob.glob("*{}*".format(out_file_name))
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name), mode="a", header=False
                )
            else:
                data_for_plot_df.to_csv("{}.csv".format(out_file_name))
            os.chdir("..")
