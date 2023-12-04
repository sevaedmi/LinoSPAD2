"""Module for calculating and plotting the timestamp differences.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 binary data output.

This file can also be imported as a module and contains the following
functions:

    * deltas_save - unpacks the binary data, calculates timestamp
    differences and saves into a '.csv' file. Works with firmware versions
    '2208' and '2212b'.
    
    * deltas_save_double - unpacks the binary data, calculates timestamp
    differences and saves into a '.csv' file. Works with firmware versions
    '2208' and '2212b'. Analyzes data from both sensor halves/both FPGAs.

    * delta_cp - collect timestamps from a '.csv' file and plot them in
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

from LinoSPAD2.functions import unpack as f_up


def deltas_save(
    path,
    pixels: list,
    rewrite: bool,
    db_num: str,
    mb_num: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    inc_offset: bool = True,
    app_calib: bool = True,
):
    """Calculate and save timestamp differences into '.csv' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels and saves them into a '.csv' table. Works with
    firmware version 2212.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.csv' file if it already exists.
    db_num : str
        LinoSPAD2 daughterboard number.
    mb_num : str
        LinoSPAD2 motherboard (FPGA) number.
    fw_ver: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    inc_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    app_calib : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while inc_offset is set to 'False', only the TDC calibration is
        applied. The default is True.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of 'db_num',
        'mb_num', and 'fw_ver' are accepted. The first error is raised
        so that the plot does not accidentally get rewritten in the
        case no clear input was given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(fw_ver, str) is False:
        raise TypeError("'fw_ver' should be string, '2212b' or '2208'")
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(db_num, str) is False:
        raise TypeError("'db_num' should be string, either 'NL11' or 'A5'")

    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
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
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .csv in a cycle < < <\n"
    )
    if fw_ver == "2212s":
        # for transforming pixel number into TDC number + pixel
        # coordinates in that TDC
        pix_coor = np.arange(256).reshape(4, 64).T
    elif fw_ver == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Mask the hot/warm pixels
    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob("*{}_{}*".format(db_num, mb_num))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        os.chdir(path_to_back)

    # Check if 'pixels' is one or two peaks, swap their positions if
    # needed
    if isinstance(pixels[0], list) is True:
        pixels_left = pixels[0]
        pixels_right = pixels[1]
        # Check if pixels from first list are to the left of the right
        # (peaks are not mixed up)
        if pixels_left[-1] > pixels_right[0]:
            plc_hld = pixels_left
            pixels_left = pixels_right
            pixels_right = plc_hld
            del plc_hld
    elif isinstance(pixels[0], int) is True:
        pixels_left = pixels
        pixels_right = pixels

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Prepare a dictionary for output
        deltas_all = {}

        # Unpack data for the requested pixels into dictionary
        data_all = f_up.unpack_bin(
            file, db_num, mb_num, fw_ver, timestamps, inc_offset, app_calib
        )

        # Calculate and collect timestamp differences
        # for q in pixels:
        for q in pixels_left:
            # for w in pixels:
            for w in pixels_right:
                if w <= q:
                    continue
                if app_mask is True and (q in mask or w in mask):
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
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])
        # Save data as a .csv file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
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

    if (
        os.path.isfile(path + "/delta_ts_data/{}.csv".format(out_file_name))
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as {file}.csv in "
            "{path} < < <".format(
                file=out_file_name,
                path=path + "\delta_ts_data",
            )
        )
    else:
        print("File wasn't generated. Check input parameters.")


def delta_save_double(
    path,
    pixels: list,
    rewrite: bool,
    db_num: str,
    mb_num1: str,
    mb_num2: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    inc_offset: bool = True,
    app_calib: bool = True,
):
    """Calculate and save timestamp differences into '.csv' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels and saves them into a '.csv' table. Works with
    firmware version 2212. Analyzes data from both sensor halves/both
    FPGAs, hence the two input parameters for LinoSPAD2 motherboards.

    Parameters
    ----------
    path : str
        Path to where two folders with data from both motherboards
        are. The folders should be named after the motherboards.
    pixels : list
        List of two pixels, one from each sensor half.
    rewrite : bool
        Switch for rewriting the '.csv' file if it already exists.
    db_num : str
        LinoSPAD2 daughterboard number.
    mb_num : str
        First LinoSPAD2 motherboard (FPGA) number.
    mb_num2 : str
        Second LinoSPAD2 motherboard (FPGA) number.
    fw_ver: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    inc_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    app_calib : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while inc_offset is set to 'False', only the TDC calibration is
        applied. The default is True.

    Raises
    ------
    FileNotFoundError
        Raised if data from the first LinoSPAD2 motherboard were not
        found.
    FileNotFoundError
        Raised if data from the second LinoSPAD2 motherboard were not
        found.
    """
    os.chdir(path)

    try:
        os.chdir("{}".format(mb_num1))
    except FileNotFoundError:
        raise FileNotFoundError("Data from {} not found".format(mb_num1))
    files_all1 = glob.glob("*.dat*")
    out_file_name = files_all1[0][:-4]
    os.chdir("..")
    try:
        os.chdir("{}".format(mb_num2))
    except FileNotFoundError:
        raise FileNotFoundError("Data from {} not found".format(mb_num2))
    files_all2 = glob.glob("*.dat*")
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]
    os.chdir("..")

    if fw_ver == "2212s":
        # for transforming pixel number into TDC number + pixel
        # coordinates in that TDC
        pix_coor = np.arange(256).reshape(4, 64).T
    elif fw_ver == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

        # check if plot exists and if it should be rewrited
    try:
        os.chdir("results/delta_t")
        if os.path.isfile(
            "{name}_delta_t_grid.png".format(name=out_file_name)
        ):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already "
                    "exists and will be rewritten ! ! !\n"
                )
            else:
                sys.exit(
                    "\nPlot already exists, 'rewrite' set to 'False', exiting."
                )
        os.chdir("../..")
    except FileNotFoundError:
        pass

    # # Mask the hot/warm pixels
    # if app_mask is True:
    #     path_to_back = os.getcwd()
    #     path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    #     os.chdir(path_to_mask)
    #     file_mask1 = glob.glob("*{}_{}*".format(db_num, mb_num1))[0]
    #     mask1 = np.genfromtxt(file_mask1).astype(int)
    #     file_mask2 = glob.glob("*{}_{}*".format(db_num, mb_num2))[0]
    #     mask2 = np.genfromtxt(file_mask2).astype(int)
    #     os.chdir(path_to_back)

    for i in tqdm(range(ceil(len(files_all1))), desc="Collecting data"):
        deltas_all = {}
        # First board
        os.chdir("{}".format(mb_num1))
        file = files_all1[i]
        data_all1 = f_up.unpack_bin(
            file, db_num, mb_num1, fw_ver, timestamps, inc_offset, app_calib
        )
        cycle_ends1 = np.where(data_all1[0].T[1] == -2)[0]
        cyc1 = np.argmin(
            np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min())
        )
        if cycle_ends1[cyc1] > np.where(data_all1[:].T[1] > 0)[0].min():
            cycle_start1 = cycle_ends1[cyc1 - 1]
        else:
            cycle_start1 = cycle_ends1[cyc1]

        os.chdir("..")

        # Second board
        os.chdir("{}".format(mb_num2))
        file = files_all2[i]
        data_all2 = f_up.unpack_bin(
            file, db_num, mb_num2, fw_ver, timestamps, inc_offset, app_calib
        )
        cycle_ends2 = np.where(data_all2[0].T[1] == -2)[0]
        cyc2 = np.argmin(
            np.abs(cycle_ends2 - np.where(data_all2[:].T[1] > 0)[0].min())
        )
        if cycle_ends2[cyc2] > np.where(data_all2[:].T[1] > 0)[0].min():
            cycle_start2 = cycle_ends2[cyc2 - 1]
        else:
            cycle_start2 = cycle_ends2[cyc2]

        os.chdir("..")

        deltas_all["{},{}".format(pixels[0], pixels[1])] = []
        tdc1, pix_c1 = np.argwhere(pix_coor == pixels[0])[0]
        pix1 = np.where(data_all1[tdc1].T[0] == pix_c1)[0]
        tdc2, pix_c2 = np.argwhere(pix_coor == pixels[1])[0]
        pix2 = np.where(data_all2[tdc2].T[0] == pix_c2)[0]

        if cycle_start1 > cycle_start2:
            cyc = len(data_all1[0].T[1]) - cycle_start1 + cycle_start2
            cycle_ends1 = cycle_ends1[cycle_ends1 > cycle_start1]
            cycle_ends2 = np.intersect1d(
                cycle_ends2[cycle_ends2 > cycle_start2],
                cycle_ends2[cycle_ends2 < cyc],
            )
        else:
            cyc = len(data_all1[0].T[1]) - cycle_start2 + cycle_start1
            cycle_ends2 = cycle_ends2[cycle_ends2 > cycle_start2]
            cycle_ends1 = np.intersect1d(
                cycle_ends1[cycle_ends1 > cycle_start1],
                cycle_ends1[cycle_ends1 < cyc],
            )
        # get timestamp for both pixels in the given cycle
        for cyc in range(len(cycle_ends1) - 1):
            pix1_ = pix1[
                np.logical_and(
                    pix1 > cycle_ends1[cyc], pix1 < cycle_ends1[cyc + 1]
                )
            ]
            if not np.any(pix1_):
                continue
            pix2_ = pix2[
                np.logical_and(
                    pix2 > cycle_ends2[cyc], pix2 < cycle_ends2[cyc + 1]
                )
            ]

            if not np.any(pix2_):
                continue
            # calculate delta t
            tmsp1 = data_all1[tdc1].T[1][
                pix1_[np.where(data_all1[tdc1].T[1][pix1_] > 0)[0]]
            ]
            tmsp2 = data_all2[tdc2].T[1][
                pix2_[np.where(data_all2[tdc2].T[1][pix2_] > 0)[0]]
            ]
            for t1 in tmsp1:
                deltas = tmsp2 - t1
                ind = np.where(np.abs(deltas) < delta_window)[0]
                deltas_all["{},{}".format(pixels[0], pixels[1])].extend(
                    deltas[ind]
                )
        # Save data as a .csv file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
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

    if (
        os.path.isfile(path + "/delta_ts_data/{}.csv".format(out_file_name))
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as {file}.csv in "
            "{path} < < <".format(
                file=out_file_name,
                path=path + "\delta_ts_data",
            )
        )
    else:
        print("File wasn't generated. Check input parameters.")


def delta_cp(
    path,
    pixels,
    rewrite: bool,
    range_left: int = -10e3,
    range_right: int = 10e3,
    step: int = 1,
    same_y: bool = False,
    color: str = "salmon",
    synchro: bool = False,
):
    """Collect and plot timestamp differences from a '.csv' file.

    Plots timestamp differences from a '.csv' file as a grid of histograms
    and as a single plot. The plot is saved in the 'results/delta_t' folder,
    which is created (if it does not already exist) in the same folder
    where data are.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences
        should be plotted.
    rewrite : bool
        Switch for rewriting the plot if it already exists.
    range_left : int, optional
        Lower limit for timestamp differences, lower values are not used.
        The default is -10e3.
    range_right : int, optional
        Upper limit for timestamp differences, higher values are not used.
        The default is 10e3.
    step : int, optional
        Histogram binning multiplier. The default is 1.
    same_y : bool, optional
        Switch for plotting the histograms with the same y-axis.
        The default is False.
    color : str, optional
        Color for the plot. The default is 'salmon'.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' are accepted. The error is
        raised so that the plot does not accidentally get rewritten.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    plt.ioff()
    os.chdir(path)
    if synchro is False:
        files_all = glob.glob("*.dat*")
        csv_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]
    else:
        folders = glob.glob("*#*")
        os.chdir(folders[0])
        files_all = glob.glob("*.dat*")
        csv_file_name = files_all[0][:-4] + "-"
        os.chdir("../{}".format(folders[1]))
        files_all = glob.glob("*.dat*")
        csv_file_name += files_all[-1][:-4]
        os.chdir("..")

    # check if plot exists and if it should be rewrited
    try:
        os.chdir("results/delta_t")
        if os.path.isfile(
            "{name}_delta_t_grid.png".format(name=csv_file_name)
        ):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already "
                    "exists and will be rewritten ! ! !\n"
                )
            else:
                sys.exit(
                    "\nPlot already exists, 'rewrite' set to 'False', exiting."
                )
        os.chdir("../..")
    except FileNotFoundError:
        pass

    print(
        "\n> > > Plotting timestamps differences as a grid of histograms < < <"
    )

    plt.rcParams.update({"font.size": 22})

    if len(pixels) > 2:
        fig, axs = plt.subplots(
            len(pixels) - 1,
            len(pixels) - 1,
            figsize=(5.5 * len(pixels), 5.5 * len(pixels)),
        )
        for ax in axs:
            for x in ax:
                x.axes.set_axis_off()
    else:
        fig = plt.figure(figsize=(14, 14))

    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    for q in tqdm(range(len(pixels)), desc="Row in plot"):
        for w in range(len(pixels)):
            if w <= q:
                continue
            if len(pixels) > 2:
                axs[q][w - 1].axes.set_axis_on()
            try:
                # keep only the required column in memory
                data_to_plot = pd.read_csv(
                    "delta_ts_data/{}.csv".format(csv_file_name),
                    usecols=["{},{}".format(pixels[q], pixels[w])],
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
                bins = np.arange(
                    np.min(data_to_plot),
                    np.max(data_to_plot),
                    17.857 * step,
                )
            except ValueError:
                print(
                    "\nCouldn't calculate bins for {q}-{w} pair: probably not "
                    "enough delta ts.".format(q=q, w=w)
                )
                continue

            if len(pixels) > 2:
                axs[q][w - 1].set_xlabel("\u0394t [ps]")
                axs[q][w - 1].set_ylabel("# of coincidences [-]")
                n, b, p = axs[q][w - 1].hist(
                    data_to_plot,
                    bins=bins,
                    color=color,
                )
            else:
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("# of coincidences [-]")
                n, b, p = plt.hist(
                    data_to_plot,
                    bins=bins,
                    color=color,
                )

            try:
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                peak_max = np.sum(n[peak_max_pos - win : peak_max_pos + win])
            except ValueError:
                peak_max = 0

            if same_y is True:
                try:
                    y_max = np.max(n)
                except ValueError:
                    y_max = 0
                    print("\nCould not find maximum y value\n")
                if y_max_all < y_max:
                    y_max_all = y_max
                if len(pixels) > 2:
                    axs[q][w - 1].set_ylim(0, y_max + 4)
                else:
                    plt.ylim(0, y_max + 4)

            if len(pixels) > 2:
                axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)
                axs[q][w - 1].set_title(
                    "Pixels {p1},{p2}\nPeak in 2 ns window: {pp}".format(
                        p1=pixels[q], p2=pixels[w], pp=int(peak_max)
                    )
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)
                if synchro is False:
                    plt.title(
                        "Pixels {p1},{p2}".format(p1=pixels[q], p2=pixels[w])
                    )
                else:
                    plt.title(
                        "Pixels {p1},{p2}".format(
                            p1=pixels[0], p2=256 + 256 - pixels[1]
                        )
                    )

            try:
                os.chdir("results/delta_t")
            except FileNotFoundError:
                os.makedirs("results/delta_t")
                os.chdir("results/delta_t")
            fig.tight_layout()  # for perfect spacing between the plots
            plt.savefig("{name}_delta_t_grid.png".format(name=csv_file_name))
            os.chdir("../..")
    print(
        "\n> > > Plot is saved as {file} in {path}< < <".format(
            file=csv_file_name + "_delta_t_grid.png",
            path=path + "/results/delta_t",
        )
    )
