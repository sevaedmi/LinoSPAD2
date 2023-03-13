"""Module with scripts for calculating and plotting the timestamp differences.

This script utilizes an unpacking module used specifically for the LinoSPAD2
binary data output.

This file can also be imported as a module and contains the following
functions:

    * plot_delta_separate - function for plotting separate figures of
    timestamp differences for each pair of pixels in the given range

    * plot_grid - function for plotting a grid of NxN plots (N+1 for number of
    pixels) of timestamp differences. Uses the calibration data. Imputing the LinoSPAD2
    board number is required.

    * plot_grid_mult - function for plotting a grid of NxN plots (N+1 for
    number of pixels) of timestamp differences. Uses the calibration data. Imputing
    the LinoSPAD2 board number is required.

    * plot_grid_mult_2212 - function for plotting a grid of NxN plots (N+1 for
    number of pixels) of timestamp differences. Works only with the 2212 version of
    the LinoSPAD2 firmware. Uses the calibration data. Imputing the LinoSPAD2
    daughterboard number is required.

"""
import glob
import os
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


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
    Plot delta t for each pair of pixels in the given range.

    Useful for debugging the LinoSPAD2 output.The plots are saved
    in the "results/delta_t/zoom" folder. In the case the folder
    does not exist, it is created automatically.

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
            "\n> > > Plotting timestamp differences, Working on {} < < <\n".format(
                filename
            )
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
                    "{filename}\nPeak position: {peak}\nPixels {p1},{p2}".format(
                        filename=filename, peak=arg_max, p1=pix[q], p2=pix[w]
                    )
                )

                try:
                    os.chdir("results/delta_t/zoom")
                except Exception:
                    os.makedirs("results/delta_t/zoom")
                    os.chdir("results/delta_t/zoom")
                plt.savefig(
                    "{name}_pixels {p1},{p2}.png".format(
                        name=filename, p1=pix[q], p2=pix[w]
                    )
                )
                plt.pause(0.1)
                plt.close()
                os.chdir("../../..")


def plot_grid(
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
    Plot a grid of delta t for all pairs of given pixels for one datafile.

    The output is saved in the "results/delta_t" folder.
    In the case the folder does not exist, it is created automatically.

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
        print("\n> > > Plotting a delta t grid, Working on {} < < <\n".format(filename))

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
                        "Pixels {p1},{p2}\nPeak position {pp}".format(
                            p1=pix[q], p2=pix[w], pp=arg_max
                        )
                    )
                else:
                    plt.xlim(range_left - 100, range_right + 100)
                    plt.title(
                        "Pixels {p1},{p2}\nPeak position {pp}".format(
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
            os.makedirs("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid_cal.png".format(name=filename))
        os.chdir("../..")


# def plot_grid_mult(
#     path,
#     pix,
#     board_number: str,
#     timestamps: int = 512,
#     range_left: float = -2.5e3,
#     range_right: float = 2.5e3,
#     show_fig: bool = False,
#     same_y: bool = False,
#     mult_files: bool = False,
# ):
#     """
#     Plot a grid of delta t for all pairs of given pixels for all datafiles.

#     Analyzes all datafiles in the given path, combining the data from each
#     and plotting a grid of timestamp differences for each pair in the given range
#     of pixels. The output is saved in the "results/delta_t" folder. In the case
#     the folder does not exist, it is created automatically.

#     Parameters
#     ----------
#     path : str
#         Path to the data file.
#     pix : array-like
#         Array of indices of pixels for analysis.
#     timestamps : int, optional
#         Number of timestamps per pixel per acquisition cycle. The default is
#         512.
#     range_left : float, optional
#         Left limit of the range for which the timestamps differences should be
#         calculated. Default is -2.5e3.
#     range_right : float, optional
#         Right limit of the range for which the timestamps differences should be
#         calculated. Default is 2.5e3.
#     show_fig : bool, optional
#         Switch for showing the output figure. The default is False.
#     same_y : bool, optional
#         Switch for setting the same ylim for all plots in the grid. The
#         default is True.
#     mult_file: bool, optional
#         Switch for processing either all data files in the directory or only
#         the last created. The default is False.

#     Returns
#     -------
#     None.

#     """

#     # check if the figure should appear in a separate window or not at all
#     if show_fig is True:
#         plt.ion()
#     else:
#         plt.ioff()

#     if mult_files is True:
#         # os.chdir(path)
#         # if len(glob.glob(".dat")) > 10:
#         #     print("Too many files.")
#         #     sys.exit()
#         print(
#             "=================================================\n"
#             "Plotting timestamp differences, Working in {}\n"
#             "=================================================".format(path)
#         )
#         data, plot_name = f_up.unpack_mult(path, board_number, timestamps)
#     else:
#         os.chdir(path)
#         files = glob.glob("*.dat*")
#         last_file = max(files, key=os.path.getctime)
#         print(
#             "=================================================\n"
#             "Plotting timestamp differences, Working on {}\n"
#             "=================================================".format(last_file)
#         )
#         data = f_up.unpack_numpy(last_file, board_number, timestamps)

#     data_pix = np.zeros((len(pix), len(data[0])))

#     for i, num in enumerate(pix):
#         data_pix[i] = data[num]
#     plt.rcParams.update({"font.size": 22})
#     if len(pix) > 2:
#         fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
#     else:
#         fig = plt.figure(figsize=(14, 14))
#     # check if the y limits of all plots should be the same
#     if same_y is True:
#         y_max_all = 0
#     print("\n> > > Calculating the timestamp differences < < <\n")
#     for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
#         for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
#             if w <= q:
#                 continue
#             data_pair = np.vstack((data_pix[q], data_pix[w]))

#             delta_ts = cd(
#                 data_pair,
#                 timestamps=timestamps,
#                 range_left=range_left,
#                 range_right=range_right,
#             )

#             if "Ne" and "540" in path:
#                 chosen_color = "seagreen"
#             elif "Ne" and "656" in path:
#                 chosen_color = "orangered"
#             elif "Ne" and "585" in path:
#                 chosen_color = "goldenrod"
#             elif "Ar" in path:
#                 chosen_color = "mediumslateblue"
#             else:
#                 chosen_color = "salmon"
#             try:
#                 bins = np.linspace(np.min(delta_ts), np.max(delta_ts), 100)
#             except Exception:
#                 print("Couldn't calculate bins: probably not enough delta ts.")
#                 continue
#             if len(pix) > 2:
#                 axs[q][w - 1].set_xlabel("\u0394t [ps]")
#                 axs[q][w - 1].set_ylabel("Timestamps [-]")
#                 n, b, p = axs[q][w - 1].hist(delta_ts, bins=bins, color=chosen_color)
#             else:
#                 plt.xlabel("\u0394t [ps]")
#                 plt.ylabel("Timestamps [-]")
#                 n, b, p = plt.hist(delta_ts, bins=bins, color=chosen_color)
#             # find position of the histogram peak
#             try:
#                 n_max = np.argmax(n)
#                 arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
#             except Exception:
#                 arg_max = None
#             if same_y is True:
#                 try:
#                     y_max = np.max(n)
#                 except ValueError:
#                     y_max = 0
#                     print("\nCould not find maximum y value\n")
#                 if y_max_all < y_max:
#                     y_max_all = y_max
#                 if len(pix) > 2:
#                     axs[q][w - 1].set_ylim(0, y_max + 4)
#                 else:
#                     plt.ylim(0, y_max + 4)
#             if len(pix) > 2:
#                 axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)

#                 axs[q][w - 1].set_title(
#                     "Pixels {p1},{p2}\nPeak position {pp}".format(
#                         p1=pix[q], p2=pix[w], pp=arg_max
#                     )
#                 )
#             else:
#                 plt.xlim(range_left - 100, range_right + 100)
#                 plt.title("Pixels {p1},{p2}".format(p1=pix[q], p2=pix[w]))
#     if same_y is True:
#         for q in range(len(pix)):
#             for w in range(len(pix)):
#                 if w <= q:
#                     continue
#                 if len(pix) > 2:
#                     axs[q][w - 1].set_ylim(0, y_max_all + 10)
#                 else:
#                     plt.ylim(0, y_max_all + 10)
#     try:
#         os.chdir("results/delta_t")
#     except FileNotFoundError:
#         os.makedirs("results/delta_t")
#         os.chdir("results/delta_t")
#     fig.tight_layout()  # for perfect spacing between the plots
#     if mult_files is True:
#         plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
#     else:
#         plt.savefig("{name}_delta_t_grid_cal.png".format(name=last_file))
#     os.chdir("../..")


def plot_grid_mult_nc(
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

    # Collect the data for the required pixels
    print("\n> > > Collecting data for the requested pixels < < <\n")
    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        files_cut = files_all[i : (i + 1)]

        data_pix = f_up.unpack_mult_cut_nc(files_cut, pix, board_number, timestamps)

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
    if len(pix) > 2:
        fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
    else:
        fig = plt.figure(figsize=(14, 14))
    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    # Calculate delta ts and plot them
    print("\n> > > Calculating the timestamp differences < < <\n")
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
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                peak_max_1 = np.sum(n[peak_max_pos - win : peak_max_pos + win])
            except Exception:
                peak_max_1 = None
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
                        p1=pix[q], p2=pix[w], pm=int(peak_max_1)
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
            plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
            os.chdir("../..")


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

    plt.rcParams.update({"font.size": 22})
    if len(pix) > 2:
        fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
    else:
        fig = plt.figure(figsize=(14, 14))
    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    # Calculate delta ts and plot them
    print("\n> > > Calculating the timestamp differences < < <\n")
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
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                peak_max_1 = np.sum(n[peak_max_pos - win : peak_max_pos + win])
            except Exception:
                peak_max_1 = None
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
                        p1=pix[q], p2=pix[w], pm=int(peak_max_1)
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
            plt.savefig("{name}_delta_t_grid_cal.png".format(name=plot_name))
            os.chdir("../..")


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
    """
    Plot a grid of delta ts for all pairs of given pixels.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : array-like
        Array of pixel numbers for which the timestamp differences should be
        calculated and plotted.
    board_number : str
        The LinoSPAD2 daughterboard nubmer.
    fw_ver : str
        The version of the 2212 firmware: 'block' or 'skip'.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.
    range_left : float, optional
        Lower limit for timestamp differences. The default is -2.5e3.
    range_right : float, optional
        Upper limit for timestamp differences. The default is 2.5e3.
    show_fig : bool, optional
        Switch for showing figure after plotting. The default is False.
    same_y : bool, optional
        Switch for making the y axis the same for all plots. The default is True.

    Returns
    -------
    None.

    """
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

                for ii, tmsp1 in enumerate(data_pix["{}".format(pix[q])]):
                    # '-1' indicate an end of a cycle
                    if tmsp1 == -1:
                        cycle += 1
                        continue
                    for jj in range(cyc2[cycle] + 1, cyc2[cycle + 1]):
                        tmsp2 = data_pix["{}".format(pix[w])][jj]
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
