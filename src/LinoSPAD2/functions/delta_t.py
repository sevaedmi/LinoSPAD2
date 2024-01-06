"""Module for calculating and plotting the timestamp differences.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 binary data output.

This file can also be imported as a module and contains the following
functions:

    * calculate_and_save_timestamp_differences - unpacks the binary data,
    calculates timestamp differences, and saves into a '.feather' file.
    Works with firmware versions '2208' and '2212b'.
    
    * calculate_and_save_timestamp_differences_full_sensor - unpacks the
    binary data, calculates timestamp differences and saves into a
    '.feather' file. Works with firmware versions '2208', '2212s' and
    '2212b'. Analyzes data from both sensor halves/both FPGAs.

    * collect_and_plot_timestamp_differences - collect timestamps from a
    '.feather' file and plot them in a grid.
    
    * collect_and_plot_timestamp_differences_full_sensor - collect
    timestamps from a '.feather' file and plot histograms of them
    in a grid. This function should be used for the full sensor
    setup.
"""
import glob
import os
import sys
from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from tqdm import tqdm

from LinoSPAD2.functions import calc_diff as cd
from LinoSPAD2.functions import unpack as f_up
from LinoSPAD2.functions import utils


def calculate_and_save_timestamp_differences(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = True,
    apply_calibration: bool = True,
):
    """Calculate and save timestamp differences into '.feather' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels, and saves them into a '.feather' table. Works with
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
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version: str
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
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and 'firmware_version'
        are accepted. The first error is raised so that the plot does not
        accidentally get rewritten in the case no clear input was given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    os.chdir(path)

    files_all = sorted(glob.glob("*.dat*"))

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # Check if the feather file exists and if it should be rewrited

    feather_file = f"{out_file_name}.feather"

    utils.file_rewrite_handling(feather_file, rewrite)

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .feather in a cycle < < <\n"
    )
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Mask the hot/warm pixels
    if app_mask is True:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        if isinstance(pixels[0], int) and isinstance(pixels[1], int):
            pixels = [pix for pix in pixels if pix not in mask]
        else:
            pixels[0] = [pix for pix in pixels[0] if pix not in mask]
            pixels[1] = [pix for pix in pixels[1] if pix not in mask]

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Prepare a dictionary for output
        deltas_all = {}

        # Unpack data for the requested pixels into dictionary
        data_all = f_up.unpack_binary_data(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset,
            apply_calibration,
        )

        deltas_all = cd.calculate_differences_2212(
            data_all, pixels, pix_coor, delta_window
        )

        # Save data as a .feather file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")

        # Version for saving to a csv, left for debugging purposes
        # csv_file = glob.glob("*{}.csv*".format(out_file_name))
        # if csv_file != []:
        #     data_for_plot_df.to_csv(
        #         "{}.csv".format(out_file_name),
        #         mode="a",
        #         index=False,
        #         header=False,
        #     )
        # else:
        #     data_for_plot_df.to_csv(
        #         "{}.csv".format(out_file_name), index=False
        #     )

        # Check if feather file exists
        feather_file = f"{out_file_name}.feather"
        if os.path.isfile(feather_file):
            # Load existing feather file
            existing_data = ft.read_feather(feather_file)

            # Append new data to the existing feather file
            combined_data = pd.concat(
                [existing_data, data_for_plot_df], axis=0
            )
            ft.write_feather(combined_data, feather_file)

        else:
            # Save as a new feather file
            ft.write_feather(data_for_plot_df, feather_file)
        os.chdir("..")

    # Check, if the file was created
    if (
        os.path.isfile(path + f"/delta_ts_data/{out_file_name}.feather")
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"{out_file_name}.feather in "
            f"{os.path.join(path, 'delta_ts_data')} < < <"
        )

    else:
        print("File wasn't generated. Check input parameters.")


def calculate_and_save_timestamp_differences_full_sensor(
    path,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = True,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
):  # TODO add option for collecting from more than just two pixels
    # TODO use pixel handling function for modularity (if possible)
    """Calculate and save timestamp differences into '.feather' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels and saves them into a '.feather' table. Works with
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
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number1 : str
        First LinoSPAD2 motherboard (FPGA) number.
    motherboard_number2 : str
        Second LinoSPAD2 motherboard (FPGA) number.
    firmware_version: str
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
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    absolute_timestamps : bool, optional
        Switch for unpacking data that were collected together with
        absolute timestamps. The default is False.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and 'firmware_version'
        are accepted. The first error is raised so that the plot does not
        accidentally get rewritten in the case no clear input was given.
    FileNotFoundError
        Raised if data from the first LinoSPAD2 motherboard were not
        found.
    FileNotFoundError
        Raised if data from the second LinoSPAD2 motherboard were not
        found.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or" "'2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    os.chdir(path)

    # Check the data from the first FPGA board
    try:
        os.chdir(f"{motherboard_number1}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data from {motherboard_number1} not found"
        ) from exc
    files_all1 = sorted(glob.glob("*.dat*"))
    out_file_name = files_all1[0][:-4]
    os.chdir("..")

    # Check the data from the second FPGA board
    try:
        os.chdir(f"{motherboard_number2}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data from {motherboard_number2} not found"
        ) from exc
    files_all2 = sorted(glob.glob("*.dat*"))
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]
    os.chdir("..")

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Check if '.feather' file with timestamps differences already
    # exists
    feather_file = f"{out_file_name}.feather"

    utils.file_rewrite_handling(feather_file, rewrite)

    # TODO add check for masked/noisy pixels
    # if app_mask is True:
    #     path_to_back = os.getcwd()
    #     path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    #     os.chdir(path_to_mask)
    #     file_mask1 = glob.glob("*{}_{}*".format(daughterboard_number, motherboard_number1))[0]
    #     mask1 = np.genfromtxt(file_mask1).astype(int)
    #     file_mask2 = glob.glob("*{}_{}*".format(daughterboard_number, motherboard_number2))[0]
    #     mask2 = np.genfromtxt(file_mask2).astype(int)
    #     os.chdir(path_to_back)

    for i in tqdm(range(ceil(len(files_all1))), desc="Collecting data"):
        deltas_all = {}
        # First board, unpack data
        os.chdir(f"{motherboard_number1}")
        file = files_all1[i]
        if not absolute_timestamps:
            data_all1 = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number1,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            (
                data_all1,
                abs_tmsp1,
            ) = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number1,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        # Collect indices of cycle ends (the '-2's)
        cycle_ends1 = np.where(data_all1[0].T[1] == -2)[0]
        cyc1 = np.argmin(
            np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min())
        )
        if cycle_ends1[cyc1] > np.where(data_all1[:].T[1] > 0)[0].min():
            cycle_start1 = cycle_ends1[cyc1 - 1]
        else:
            cycle_start1 = cycle_ends1[cyc1]

        os.chdir("..")

        # Second board, unpack data
        os.chdir(f"{motherboard_number2}")
        file = files_all2[i]
        if not absolute_timestamps:
            data_all2 = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number2,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            (
                data_all2,
                abs_tmsp2,
            ) = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number2,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )

        # Collect indices of cycle ends (the '-2's)
        cycle_ends2 = np.where(data_all2[0].T[1] == -2)[0]
        cyc2 = np.argmin(
            np.abs(cycle_ends2 - np.where(data_all2[:].T[1] > 0)[0].min())
        )
        if cycle_ends2[cyc2] > np.where(data_all2[:].T[1] > 0)[0].min():
            cycle_start2 = cycle_ends2[cyc2 - 1]
        else:
            cycle_start2 = cycle_ends2[cyc2]

        os.chdir("..")

        pix_left_peak = pixels[0]
        # The following piece of code take any value for the pixel from
        # the second motherboard, either in terms of full sensor (so
        # a value >=256) or in terms of single sensor half

        if pixels[1] >= 256:
            if pixels[1] > 256 + 127:
                pix_right_peak = 255 - (pixels[1] - 256)
            else:
                pix_right_peak = pixels[1] - 256 + 128
        elif pixels[1] > 127:
            pix_right_peak = 255 - pixels[1]
        else:
            pix_right_peak = pixels[1] + 128

        # Get the data from the requested pixel only
        deltas_all[f"{pix_left_peak},{pix_right_peak}"] = []
        tdc1, pix_c1 = np.argwhere(pix_coor == pix_left_peak)[0]
        pix1 = np.where(data_all1[tdc1].T[0] == pix_c1)[0]
        tdc2, pix_c2 = np.argwhere(pix_coor == pix_right_peak)[0]
        pix2 = np.where(data_all2[tdc2].T[0] == pix_c2)[0]

        # Data from one of the board should be shifted as data collection
        # on one of the board is started later
        if cycle_start1 > cycle_start2:
            cyc = len(data_all1[0].T[1]) - cycle_start1 + cycle_start2
            cycle_ends1 = cycle_ends1[cycle_ends1 >= cycle_start1]
            cycle_ends2 = np.intersect1d(
                cycle_ends2[cycle_ends2 >= cycle_start2],
                cycle_ends2[cycle_ends2 <= cyc],
            )
        else:
            cyc = len(data_all1[0].T[1]) - cycle_start2 + cycle_start1
            cycle_ends2 = cycle_ends2[cycle_ends2 >= cycle_start2]
            cycle_ends1 = np.intersect1d(
                cycle_ends1[cycle_ends1 >= cycle_start1],
                cycle_ends1[cycle_ends1 <= cyc],
            )
        # Get timestamps for both pixels in the given cycle
        for cyc in range(len(cycle_ends1) - 1):
            pix1_ = pix1[
                np.logical_and(
                    pix1 >= cycle_ends1[cyc], pix1 < cycle_ends1[cyc + 1]
                )
            ]
            if not np.any(pix1_):
                continue
            pix2_ = pix2[
                np.logical_and(
                    pix2 >= cycle_ends2[cyc], pix2 < cycle_ends2[cyc + 1]
                )
            ]

            if not np.any(pix2_):
                continue
            # Calculate delta t
            tmsp1 = data_all1[tdc1].T[1][
                pix1_[np.where(data_all1[tdc1].T[1][pix1_] > 0)[0]]
            ]
            tmsp2 = data_all2[tdc2].T[1][
                pix2_[np.where(data_all2[tdc2].T[1][pix2_] > 0)[0]]
            ]
            for t1 in tmsp1:
                deltas = tmsp2 - t1
                ind = np.where(np.abs(deltas) < delta_window)[0]
                deltas_all[f"{pix_left_peak},{pix_right_peak}"].extend(
                    deltas[ind]
                )
        # Version using csv files; left for debugging
        # # Save data as a .csv file in a cycle so data is not lost
        # # in the case of failure close to the end
        # data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        # del deltas_all
        # data_for_plot_df = data_for_plot_df.T
        # try:
        #     os.chdir("delta_ts_data")
        # except FileNotFoundError:
        #     os.mkdir("delta_ts_data")
        #     os.chdir("delta_ts_data")
        # csv_file = glob.glob("*{}.csv*".format(out_file_name))
        # if csv_file != []:
        #     data_for_plot_df.to_csv(
        #         "{}.csv".format(out_file_name),
        #         mode="a",
        #         index=False,
        #         header=False,
        #     )
        # else:
        #     data_for_plot_df.to_csv(
        #         "{}.csv".format(out_file_name), index=False
        #     )
        # os.chdir("..")

        # Save data to a feather file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T

        feather_file = f"{out_file_name}.feather"

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")

        if os.path.isfile(feather_file):
            # Load existing Feather file
            existing_data = ft.read_feather(feather_file)

            # Append new data to the existing Feather file
            combined_data = pd.concat(
                [existing_data, data_for_plot_df], axis=0
            )
            ft.write_feather(combined_data, feather_file)

        else:
            # Save as a new Feather file
            ft.write_feather(data_for_plot_df, feather_file)

        os.chdir("..")

    # Check if the file with the results was created
    if (
        os.path.isfile(
            os.path.join(path, f"/delta_ts_data/{out_file_name}.feather")
        )
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"{out_file_name}.feather in "
            f"{os.path.join(path, 'delta_ts_data')} < < <"
        )
    else:
        print("File wasn't generated. Check input parameters.")


def collect_and_plot_timestamp_differences(
    path,
    pixels,
    rewrite: bool,
    range_left: int = -10e3,
    range_right: int = 10e3,
    step: int = 1,
    same_y: bool = False,
    color: str = "salmon",
):
    """Collect and plot timestamp differences from a '.feather' file.

    Plots timestamp differences from a '.feather' file as a grid of histograms
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

    files_all = sorted(glob.glob("*.dat*"))
    csv_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # Check if plot exists and if it should be rewritten
    try:
        os.chdir("results/delta_t")
        if os.path.isfile(f"{csv_file_name}_delta_t_grid.png"):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already"
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

    # Prepare the grid for the plots based on the number of pixels
    # given
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

    # Check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    for q, _ in tqdm(enumerate(pixels), desc="Row in plot"):
        for w, _ in enumerate(pixels):
            if w <= q:
                continue
            if len(pixels) > 2:
                axs[q][w - 1].axes.set_axis_on()

            # csv-version, left for debugging
            # Read data from csv file

            # try:
            #     os.path.isfile("delta_ts_data/{}.csv".format(csv_file_name))
            #     csv_file = "delta_ts_data/{}.csv".format(csv_file_name)
            # except FileNotFoundError:
            #     try:
            #         dash_position = csv_file.find("-")
            #         csv_file_name = (
            #             "delta_ts_data/"
            #             + csv_file[dash_position + 1 : -4]
            #             + "-"
            #             + csv_file[14:dash_position]
            #             + ".csv"
            #         )
            #     except FileNotFoundError:
            #         raise FileNotFoundError(
            #             "'.csv' file with timestamps"
            #             "differences was not found"
            #         )

            # csv_file_path = "delta_ts_data/{}.csv".format(csv_file_name)
            # if os.path.isfile(csv_file_path):
            #     csv_file = csv_file_path
            # else:
            #     raise FileNotFoundError(
            #         "'.csv' file with timestamps differences was not found"
            #     )

            # try:
            #     data_to_plot = pd.read_csv(
            #         csv_file,
            #         usecols=["{},{}".format(pixels[q], pixels[w])],
            #     ).dropna()
            # except ValueError:
            #     continue

            # Read data from Feather file
            try:
                data_to_plot = ft.read_feather(
                    f"delta_ts_data/{csv_file_name}.feather",
                    columns=[f"{pixels[q]},{pixels[w]}"],
                ).dropna()
            except ValueError:
                continue

            # Prepare the data for the plot
            data_to_plot = np.array(data_to_plot)
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot < range_left)
            )
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot > range_right)
            )

            # Bins should be in units of 17.857 ps - average bin width
            # of the LinoSPAD2 TDCs
            try:
                bins = np.arange(
                    np.min(data_to_plot),
                    np.max(data_to_plot),
                    17.857 * step,
                )
            except ValueError:
                print(
                    f"\nCouldn't calculate bins for {q}-{w} pair: "
                    "probably not enough delta ts."
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

            # Find number of timestamps differences in a 2 ns window
            # around the peak (HBT or cross-talk)
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
                    f"Pixels {pixels[q]},{pixels[w]}\nPeak in 2 ns "
                    f"window: {int(peak_max)}"
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)

                plt.title(f"Pixels {pixels[q]},{pixels[w]}")

            # Save the figure
            try:
                os.chdir("results/delta_t")
            except FileNotFoundError:
                os.makedirs("results/delta_t")
                os.chdir("results/delta_t")
            # fig.tight_layout()  # for perfect spacing between the plots
            plt.savefig(f"{csv_file_name}_delta_t_grid.png")
            os.chdir("../..")

    print(
        "\n> > > Plot is saved as {file} in {path}< < <".format(
            file=csv_file_name + "_delta_t_grid.png",
            path=path + "/results/delta_t",
        )
    )


def collect_and_plot_timestamp_differences_full_sensor(
    path,
    pixels,
    rewrite: bool,
    range_left: int = -10e3,
    range_right: int = 10e3,
    step: int = 1,
    same_y: bool = False,
    color: str = "salmon",
):
    # TODO pixel_handling
    """Collect and plot timestamp differences from a '.feather' file.

    Plots timestamp differences from a '.feather' file as a grid of
    histograms and as a single plot. The plot is saved in the
    'results/delta_t' folder, which is created (if it does not already
    exist) in the same folder where data are.

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

    # Get the data files names for finding the appropriate '.feather'
    # file with timestamps differences, checking both options, depending
    # on which board was analyzed first
    folders = glob.glob("*#*")
    os.chdir(folders[0])
    files_all = sorted(glob.glob("*.dat*"))
    csv_file_name1 = files_all[0][:-4] + "-"
    csv_file_name2 = "-" + files_all[-1][:-4]
    os.chdir("../{}".format(folders[1]))
    files_all = sorted(glob.glob("*.dat*"))
    csv_file_name1 += files_all[-1][:-4]
    csv_file_name2 = files_all[0][:-4] + csv_file_name2
    os.chdir("..")

    # Check if plot exists and if it should be rewritten
    try:
        os.chdir("results/delta_t")
        if os.path.isfile(
            "{name}_delta_t_grid.png".format(name=csv_file_name1)
        ):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already"
                    "exists and will be rewritten ! ! !\n"
                )
            else:
                sys.exit(
                    "\nPlot already exists, 'rewrite' set to 'False', exiting."
                )

        elif os.path.isfile(
            "{name}_delta_t_grid.png".format(name=csv_file_name2)
        ):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already"
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
    # Prepare the grid for the plots based on the number of pixels
    # given
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

    # Check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    for q, _ in tqdm(enumerate(pixels), desc="Row in plot"):
        for w, _ in enumerate(pixels):
            if w <= q:
                continue
            if len(pixels) > 2:
                axs[q][w - 1].axes.set_axis_on()

            # csv-version, left for debugging
            # Read data from csv file

            # try:
            #     os.path.isfile("delta_ts_data/{}.csv".format(csv_file_name))
            #     csv_file = "delta_ts_data/{}.csv".format(csv_file_name)
            # except FileNotFoundError:
            #     try:
            #         dash_position = csv_file.find("-")
            #         csv_file_name = (
            #             "delta_ts_data/"
            #             + csv_file[dash_position + 1 : -4]
            #             + "-"
            #             + csv_file[14:dash_position]
            #             + ".csv"
            #         )
            #     except FileNotFoundError:
            #         raise FileNotFoundError(
            #             "'.csv' file with timestamps"
            #             "differences was not found"
            #         )

            # Check if the file with timestamps differences is there
            csv_file_path1 = "delta_ts_data/{}.feather".format(csv_file_name1)
            csv_file_path2 = "delta_ts_data/{}.feather".format(csv_file_name2)
            csv_file, csv_file_name = (
                (csv_file_path1, csv_file_name1)
                if os.path.isfile(csv_file_path1)
                else (csv_file_path2, csv_file_name2)
            )
            # print(csv_file)
            if not os.path.isfile(csv_file):
                raise FileNotFoundError(
                    "'.feather' file with timestamps differences was not found"
                )

            # csv-version, left for debugging
            # try:
            #     data_to_plot = pd.read_csv(
            #         csv_file,
            #         usecols=["{},{}".format(pixels[q], pixels[w])],
            #     ).dropna()
            # except ValueError:
            #     continue
            # Read data from Feather file
            try:
                data_to_plot = ft.read_feather(
                    "delta_ts_data/{name}.feather".format(name=csv_file_name),
                    columns=["{},{}".format(pixels[q], pixels[w])],
                ).dropna()
            except ValueError:
                continue

            # Prepare the data for the plot
            data_to_plot = np.array(data_to_plot)
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot < range_left)
            )
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot > range_right)
            )

            # Bins should be in units of 17.857 ps - average bin width
            # of the LinoSPAD2 TDCs
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

            # Find number of timestamps differences in a 2 ns window
            # around the peak (HBT or cross-talk)
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
                    f"Pixels {pixels[q]},{pixels[w]}\nPeak in 2 ns "
                    f"window: {int(peak_max)}"
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)

                plt.title(f"Pixels {pixels[0]},{256 + 255 - pixels[1]}")

            # Save the figure
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
