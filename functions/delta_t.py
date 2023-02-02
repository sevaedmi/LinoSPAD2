""" Module with scripts for calculating and plotting the timestamp differences
for different pairs of pixels.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * plot_delta_separate - function for plotting separate figures of
    timestamp differences for each pair of pixels in the given range

    * plot_grid_df - function for plotting a grid of NxN plots (N for number of
    pixels) of timestamps differences. Works with tidy dataframes; currently, the
    fastest approach.

    * plot_grid_calib - function for plotting a grid of NxN plots (N for number of
    pixels) of timestamp differences. Uses the calibration data. Imputing the LinoSPAD2
    board number is required.

    * plot_grid_calib_mult - function for plotting a grid of NxN plots (N for
    number of pixels) of timestamp differences. Uses the calibration data. Imputing
    the LinoSPAD2 board number is required. Provides an option to combine all the
    data files or analyze only the last created.

"""

import os
import glob
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from functions import unpack as f_up
from functions.calc_diff import calc_diff_df, calc_diff as cd


def compute_delta_t(pixel_0, pixel_1, timestampsnmr: int = 512, timewindow: int = 5000):

    nmr_of_cycles = int(len(pixel_0) / timestampsnmr)
    output = []

    # start = time.time()
    for cycle in range(nmr_of_cycles):
        for timestamp_pix0 in range(timestampsnmr):
            if (
                pixel_0[cycle * timestampsnmr + timestamp_pix0] == -1
                or pixel_0[cycle * timestampsnmr + timestamp_pix0] <= 1e-9
            ):
                break
            for timestamp_pix1 in range(timestampsnmr):
                if (
                    pixel_1[cycle * timestampsnmr + timestamp_pix1] == -1
                    or pixel_1[cycle * timestampsnmr + timestamp_pix1] == 0
                ):
                    break
                if (
                    np.abs(
                        pixel_0[cycle * timestampsnmr + timestamp_pix0]
                        - pixel_1[cycle * timestampsnmr + timestamp_pix1]
                    )
                    < timewindow
                ):
                    output.append(
                        pixel_0[cycle * timestampsnmr + timestamp_pix0]
                        - pixel_1[cycle * timestampsnmr + timestamp_pix1]
                    )
                else:
                    continue
    return output


def plot_delta_separate(path, pix, timestamps: int = 512):
    """
    Plots delta t for each pair of pixels in the given range.  The plots are
    saved in the "results/delta_t/zoom" folder. In the case the folder does
    not exist, it is created automatically.

    Parameters
    ----------
    path : str
        Path to data file.
    pix : array-like
        Array of indices of 5 pixels for analysis.
    timestamps : int
        Number of timestamps per acq cycle per pixel in the file. The default
        is 512.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):

        print(
            "======================================================\n"
            "Plotting timestamp differences, Working on {}\n"
            "======================================================".format(filename)
        )

        data = f_up.unpack_numpy(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]
        plt.rcParams.update({"font.size": 22})

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                data_pair = np.vstack((data_pix[q], data_pix[w]))

                delta_ts = cd(data_pair, timestamps=timestamps)

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"
                try:
                    # bins = np.arange(np.min(delta_ts), np.max(delta_ts), 17.857 * 2)
                    bins = 120
                except Exception:
                    continue
                plt.figure(figsize=(11, 7))
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n = plt.hist(delta_ts, bins=bins, color=chosen_color)[0]

                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
                except Exception:
                    arg_max = None
                plt.title(
                    "{filename}\nPeak position: {peak}\nPixels {p1}-{p2}".format(
                        filename=filename, peak=arg_max, p1=pix[q], p2=pix[w]
                    )
                )

                try:
                    os.chdir("results/delta_t/zoom")
                except Exception:
                    os.mkdir("results/delta_t/zoom")
                    os.chdir("results/delta_t/zoom")
                plt.savefig(
                    "{name}_pixels {p1}-{p2}.png".format(
                        name=filename, p1=pix[q], p2=pix[w]
                    )
                )
                plt.pause(0.1)
                plt.close()
                os.chdir("../../..")


def plot_grid_df(path, pix, timestamps: int = 512):
    """
    This function plots a grid of timestamp differences for the given range of
    pixels. This function utilizes the pandas dataframes.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : list or array-like
        Pixel numbers for which the delta ts are calculated.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, file in enumerate(DATA_FILES):

        data = f_up.unpack_binary_df(file, timestamps)

        fig, axes = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(24, 24))
        plt.suptitle("{} delta ts".format(file))

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                deltas = calc_diff_df(
                    data[data.Pixel == pix[q]], data[data.Pixel == pix[w]]
                )

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"
                try:
                    bins = np.arange(int(deltas.min()), int(deltas.max()), 17.857 * 2)
                except Exception:
                    continue
                sns.histplot(
                    ax=axes[q, w - 1],
                    x="Delta t",
                    data=deltas,
                    bins=bins,
                    color=chosen_color,
                )
                axes[q, w - 1].set_title(
                    "Pixels {pix1}-{pix2}".format(pix1=pix[q], pix2=pix[w])
                )
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.mkdir("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid_df.png".format(name=file))
        os.chdir("../..")


def plot_peak_vs_peak(path, pix1, pix2, timestamps: int = 512):
    """
    Function for calculating timestamp differences between two groups of pixels where
    the light beam falls. The differences are plotted as a histogram.

    Parameters
    ----------
    path : str
        Path to data files.
    pix1 : list or array-like
        List of pixel numbers from the first peak.
    pix2 : list or array-like
        List of pixel numbers from the second peak.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, file in enumerate(DATA_FILES):

        data = f_up.unpack_binary_df(file, timestamps)

        deltas_total = pd.DataFrame()

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix1)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix2)), desc="Subtrahend pixel"):
                deltas = calc_diff_df(
                    data[data.Pixel == pix1[q]], data[data.Pixel == pix2[w]]
                )
                deltas_total = pd.concat([deltas_total, deltas], ignore_index=True)
        if "Ne" and "540" in path:
            chosen_color = "seagreen"
        elif "Ne" and "656" in path:
            chosen_color = "orangered"
        elif "Ar" in path:
            chosen_color = "mediumslateblue"
        else:
            chosen_color = "salmon"
        try:
            bins = np.arange(
                int(deltas_total.min()), int(deltas_total.max()), 17.857 * 2
            )
        except Exception:
            continue
        fig_sns = sns.histplot(
            x="Delta t", data=deltas_total, color=chosen_color, bins=bins
        )
        fig = fig_sns.get_figure()
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.mkdir("results/delta_t")
            os.chdir("results/delta_t")
        fig.savefig("{name}_peak_v_peak_df.png".format(name=file))
        os.chdir("../..")


def plot_grid_calib(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = True,
):
    """
    Plots a grid of delta t for different pairs of pixels for the
    pixels in the given range. The output is saved in the "results/delta_t"
    folder. In the case the folder does not exist, it is created automatically.


    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of pixels for analysis.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    range_left : float, optional
        Left limit of the range for which the timestamps differences should be
        calculated. Default is -2.5e3.
    range_right : float, optional
        Right limit of the range for which the timestamps differences should be
        calculated. Default is 2.5e3.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    same_y : bool, optional
        Switch for setting the same ylim for all plots in the grid. The
        default is True.

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

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):

        print(
            "=====================================================\n"
            "Plotting a delta t grid, Working on {}\n"
            "=====================================================\n".format(filename)
        )

        data = f_up.unpack_calib(filename, board_number, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]
        plt.rcParams.update({"font.size": 22})
        if len(pix) > 2:
            fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
        else:
            fig = plt.figure(figsize=(14, 14))
        # check if the y limits of all plots should be the same
        if same_y is True:
            y_max_all = 0
        print("\n> > > Calculating the timestamp differences < < <\n")
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
                    bins = np.linspace(np.min(delta_ts), np.max(delta_ts), 100)
                except Exception:
                    print("Couldn't calculate bins: probably not enough delta ts.")
                    continue
                if len(pix) > 2:
                    axs[q][w - 1].set_xlabel("\u0394t [ps]")
                    axs[q][w - 1].set_ylabel("Timestamps [-]")
                    n, b, p = axs[q][w - 1].hist(
                        delta_ts, bins=bins, color=chosen_color
                    )
                else:
                    plt.xlabel("\u0394t [ps]")
                    plt.ylabel("Timestamps [-]")
                    n, b, p = plt.hist(delta_ts, bins=bins, color=chosen_color)
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
                    plt.title(
                        "Pixels {p1}-{p2}\nPeak position {pp}".format(
                            p1=pix[q], p2=pix[w], pp=arg_max
                        )
                    )
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
            os.mkdir("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=filename))
        os.chdir("../..")


def plot_grid_calib_mult(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = True,
    mult_files: bool = False,
):
    """
    Plots a grid of delta t for different pairs of pixels for the
    pixels in the given range. The output is saved in the "results/delta_t"
    folder. In the case the folder does not exist, it is created automatically.
    Combines all ".dat" files in the folder or analyses only the last file created.

    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of pixels for analysis.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    range_left : float, optional
        Left limit of the range for which the timestamps differences should be
        calculated. Default is -2.5e3.
    range_right : float, optional
        Right limit of the range for which the timestamps differences should be
        calculated. Default is 2.5e3.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    same_y : bool, optional
        Switch for setting the same ylim for all plots in the grid. The
        default is True.
    mult_file: bool, optional
        Switch for processing either all data files in the directory or only
        the last created. The default is False.

    Returns
    -------
    None.

    """

    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    if mult_files is True:
        # os.chdir(path)
        # if len(glob.glob(".dat")) > 10:
        #     print("Too many files.")
        #     sys.exit()
        print(
            "=================================================\n"
            "Plotting timestamp differences, Working in {}\n"
            "=================================================".format(path)
        )
        data, plot_name = f_up.unpack_calib_mult(path, board_number, timestamps)
    else:
        os.chdir(path)
        files = glob.glob("*.dat*")
        last_file = max(files, key=os.path.getctime)
        print(
            "=================================================\n"
            "Plotting timestamp differences, Working on {}\n"
            "=================================================".format(last_file)
        )
        data = f_up.unpack_calib(last_file, board_number, timestamps)

    data_pix = np.zeros((len(pix), len(data[0])))

    for i, num in enumerate(pix):
        data_pix[i] = data[num]
    plt.rcParams.update({"font.size": 22})
    if len(pix) > 2:
        fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
    else:
        fig = plt.figure(figsize=(14, 14))
    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0
    print("\n> > > Calculating the timestamp differences < < <\n")
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
                bins = np.linspace(np.min(delta_ts), np.max(delta_ts), 100)
            except Exception:
                print("Couldn't calculate bins: probably not enough delta ts.")
                continue
            if len(pix) > 2:
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("Timestamps [-]")
                n, b, p = axs[q][w - 1].hist(delta_ts, bins=bins, color=chosen_color)
            else:
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n, b, p = plt.hist(delta_ts, bins=bins, color=chosen_color)
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
        os.mkdir("results/delta_t")
        os.chdir("results/delta_t")
    fig.tight_layout()  # for perfect spacing between the plots
    if mult_files is True:
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
    else:
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=last_file))
    os.chdir("../..")


def plot_grid_calib_mult_cut(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = True,
    mult_files: bool = False,
):
    """
    Plots a grid of delta t for different pairs of pixels for the
    pixels in the given range. The output is saved in the "results/delta_t"
    folder. In the case the folder does not exist, it is created automatically.
    Combines all ".dat" files in the folder or analyses only the last file created.
    Memory-friendly version as it unpacks only the requested pixels (in the case
    of multiple files).

    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of pixels for analysis.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    range_left : float, optional
        Left limit of the range for which the timestamps differences should be
        calculated. Default is -2.5e3.
    range_right : float, optional
        Right limit of the range for which the timestamps differences should be
        calculated. Default is 2.5e3.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    same_y : bool, optional
        Switch for setting the same ylim for all plots in the grid. The
        default is True.
    mult_file: bool, optional
        Switch for processing either all data files in the directory or only
        the last created. The default is False.

    Returns
    -------
    None.

    """

    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    if mult_files is True:
        print(
            "=================================================\n"
            "Plotting timestamp differences, Working in {}\n"
            "=================================================".format(path)
        )
        data_pix, plot_name = f_up.unpack_calib_mult_cut(
            path, pix, board_number, timestamps
        )
    else:
        os.chdir(path)
        files = glob.glob("*.dat*")
        last_file = max(files, key=os.path.getctime)
        print(
            "=================================================\n"
            "Plotting timestamp differences, Working on {}\n"
            "=================================================".format(last_file)
        )
        data = f_up.unpack_calib(last_file, board_number, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num in enumerate(pix):
            data_pix[i] = data[num]

    plt.rcParams.update({"font.size": 22})
    if len(pix) > 2:
        fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
    else:
        fig = plt.figure(figsize=(14, 14))
    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0
    print("\n> > > Calculating the timestamp differences < < <\n")
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
                bins = np.linspace(np.min(delta_ts), np.max(delta_ts), 100)
            except Exception:
                print("Couldn't calculate bins: probably not enough delta ts.")
                continue
            if len(pix) > 2:
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("Timestamps [-]")
                n, b, p = axs[q][w - 1].hist(delta_ts, bins=bins, color=chosen_color)
            else:
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n, b, p = plt.hist(delta_ts, bins=bins, color=chosen_color)
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
        os.mkdir("results/delta_t")
        os.chdir("results/delta_t")
    fig.tight_layout()  # for perfect spacing between the plots
    if mult_files is True:
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
    else:
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=last_file))
    os.chdir("../..")
