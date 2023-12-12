"""Module with scripts for sharing (and plotting) data in a compact form.

The module provides a simple and memory-friendly way of sharing data - 
that is processed '.dat' data in a form of a '.txt' file with number of
timestamps collected by each pixel and a '.csv' file with timestamp
differences. Additionally, function for plotting both the sensor
population plot and the delta t histograms are provided.

This file can also be imported as a module and contains the following
functions:

    * compact_share - unpacks all '.dat' files in the given folder,
    collects the number of timestamps in each pixel and packs it into a
    '.txt' file, calculates timestamp differences and packs them into a 
    '.csv' file.
    
    * plot_shared - plots the sensor population plot from the '.txt'
    file.
    
    * delta_cp_shared - plots the delta t histograms from the '.csv'
    file.

"""


import glob
import os
import sys
import time
from math import ceil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from matplotlib import pyplot as plt
from tqdm import tqdm

from LinoSPAD2.functions import unpack as f_up


def compact_share(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    delta_window: float = 50e3,
    include_offset: bool = True,
    apply_calibration: bool = True,
):
    """Collect delta ts and sensor population to a zip file.

    Unpacks data in the given folder, collects all timestamp differences
    and sensor population, saving the first one to a '.csv' file and
    the second to a '.txt' one, zipping both for compact output ready
    to share.

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
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of 'firmware_version'
        are accepted. First error is raised so that the plot does not
        accidentally gets rewritten in the case no clear input was
        given.

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
            "'firmware_version' should be string, '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError(
            "'daughterboard_number' should be string, either 'NL11' or 'A5'"
        )
    if isinstance(motherboard_number, str) is False:
        raise TypeError("'motherboard_number' should be string")

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
                print(
                    "\n csv file already exists, 'rewrite' set to"
                    "'False', passing"
                )
                pass
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    if rewrite is True:
        print(
            "\n> > > Collecting data for delta t plot for the requested "
            "pixels and saving it to .csv in a cycle < < <\n"
        )
        if firmware_version == "2212s":
            # for transforming pixel number into TDC number + pixel
            # coordinates in that TDC
            pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

        # Mask the hot/warm pixels
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob(
            "*{}_{}*".format(daughterboard_number, motherboard_number)
        )[0]
        mask = np.genfromtxt(file_mask).astype(int)
        os.chdir(path_to_back)

        # Check if 'pixels' is one or two peaks, swap their positions if
        # needed
        if isinstance(pixels[0], list) is True:
            pixels_left = sorted(pixels[0])
            pixels_right = sorted(pixels[1])
            # Check if pixels from first list are to the left of the right
            # (peaks are not mixed up)
            if pixels_left[-1] > pixels_right[0]:
                plc_hld = pixels_left
                pixels_left = pixels_right
                pixels_right = plc_hld
                del plc_hld
        elif isinstance(pixels[0], int) is True:
            pixels.sort()
            pixels_left = pixels
            pixels_right = pixels

        # Prepare array for sensor population
        valid_per_pixel = np.zeros(256)

        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}

            # Unpack data for the requested pixels into dictionary
            data_all = f_up.unpack_bin(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )

            # Calculate and collect timestamp differences
            # for q in pixels:
            for q in pixels_left:
                # for w in pixels:
                for w in pixels_right:
                    if w <= q:
                        continue
                    if q in mask or w in mask:
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
            # Collect sensor population
            for k in range(256):
                tdc, pix = np.argwhere(pix_coor == k)[0]
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                ind1 = np.where(data_all[tdc].T[1][ind] > 0)[0]
                valid_per_pixel[k] += len(data_all[tdc].T[1][ind[ind1]])

            # Save data as a .csv file in a cycle so data is not lost
            # in the case of failure close to the end
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("compact_share")
            except FileNotFoundError:
                os.mkdir("compact_share")
                os.chdir("compact_share")
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

        os.chdir("compact_share")
        np.savetxt("sen_pop_{}.txt".format(out_file_name), valid_per_pixel)
        # Create a ZipFile Object
        with ZipFile("{}.zip".format(out_file_name), "w") as zip_object:
            # Adding files that need to be zipped
            zip_object.write("{}.csv".format(out_file_name))
            zip_object.write("sen_pop_{}.txt".format(out_file_name))

            print(
                "\n> > > Timestamp differences are saved as {file}.csv and "
                "sensor population as sen_pop.txt in "
                "{path} < < <".format(
                    file=out_file_name,
                    path=path + "\delta_ts_data",
                )
            )


# TODO check the feather file, seems to be close to empty
def compact_share_feather(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    delta_window: float = 50e3,
    include_offset: bool = True,
    apply_calibration: bool = True,
):
    """Collect delta timestamp differences and sensor population, and
    save to a feather file.

    Unpacks data in the given folder, calculates timestamp differences
    and sensor population for the specified pixels, saving the timestamp
    differences to a '.feather' file and the sensor population to a '.txt'
    file. Both files are then zipped for compact output ready to share.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved, or list of two lists with pixel numbers
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
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'firmware_version' are accepted. The first error is raised so
        that the files are not
        accidentally overwritten in the case of unclear input.

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
            "'firmware_version' should be string, '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError(
            "'daughterboard_number' should be string, either 'NL11' or 'A5'"
        )
    if isinstance(motherboard_number, str) is False:
        raise TypeError("'motherboard_number' should be string")

    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if feather file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        feather_file = "{}.feather".format(out_file_name)
        if os.path.isfile(feather_file):
            if rewrite is True:
                print(
                    "\n! ! ! feather file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
                )
                for i in range(5):
                    print(
                        "\n! ! ! Deleting the file in {} ! ! !\n".format(5 - i)
                    )
                    time.sleep(1)
                os.remove(feather_file)
            else:
                print(
                    "\n feather file already exists, 'rewrite' set to"
                    "'False', passing"
                )
                pass
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    if rewrite is True:
        print(
            "\n> > > Collecting data for delta t plot for the requested "
            "pixels and saving it to .feather in a cycle < < <\n"
        )
        if firmware_version == "2212s":
            # for transforming pixel number into TDC number + pixel
            # coordinates in that TDC
            pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

        # Mask the hot/warm pixels
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob(
            "*{}_{}*".format(daughterboard_number, motherboard_number)
        )[0]
        mask = np.genfromtxt(file_mask).astype(int)
        os.chdir(path_to_back)

        # Check if 'pixels' is one or two peaks, swap their positions if
        # needed
        if isinstance(pixels[0], list) is True:
            pixels_left = sorted(pixels[0])
            pixels_right = sorted(pixels[1])
            # Check if pixels from the first list are to the left of the right
            # (peaks are not mixed up)
            if pixels_left[-1] > pixels_right[0]:
                plc_hld = pixels_left
                pixels_left = pixels_right
                pixels_right = plc_hld
                del plc_hld
        elif isinstance(pixels[0], int) is True:
            pixels.sort()
            pixels_left = pixels
            pixels_right = pixels

        # Prepare array for sensor population
        valid_per_pixel = np.zeros(256)

        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}

            # Unpack data for the requested pixels into dictionary
            data_all = f_up.unpack_bin(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )

            # Calculate and collect timestamp differences
            for q in pixels_left:
                for w in pixels_right:
                    if w <= q:
                        continue
                    if q in mask or w in mask:
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
            # Collect sensor population
            for k in range(256):
                tdc, pix = np.argwhere(pix_coor == k)[0]
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                ind1 = np.where(data_all[tdc].T[1][ind] > 0)[0]
                valid_per_pixel[k] += len(data_all[tdc].T[1][ind[ind1]])

            # Save data as a .feather file in a cycle so data is not lost
            # in the case of failure close to the end
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("compact_share")
            except FileNotFoundError:
                os.mkdir("compact_share")
                os.chdir("compact_share")
            feather_file = "{}.feather".format(out_file_name)
            if os.path.isfile(feather_file):
                existing_data = feather.read_feather(feather_file)
                combined_data = pd.concat(
                    [existing_data, data_for_plot_df], axis=0
                )
                feather.write_feather(combined_data, feather_file)
            else:
                feather.write_feather(data_for_plot_df, feather_file)
            os.chdir("..")

        os.chdir("compact_share")
        np.savetxt("sen_pop_{}.txt".format(out_file_name), valid_per_pixel)
        # Create a ZipFile Object
        with ZipFile("{}.zip".format(out_file_name), "w") as zip_object:
            # Adding files that need to be zipped
            zip_object.write("{}.feather".format(out_file_name))
            zip_object.write("sen_pop_{}.txt".format(out_file_name))

            print(
                "\n> > > Timestamp differences are saved as {file}.feather and "
                "sensor population as sen_pop.txt in "
                "{path} < < <".format(
                    file=out_file_name,
                    path=path + "\delta_ts_data",
                )
            )


def plot_shared(
    path,
    daughterboard_number: str,
    motherboard_number: str,
    show_fig: bool = False,
    app_mask: bool = True,
    color: str = "salmon",
):
    """Plots sensor population from a '.txt' file.

    Plots the sensor population plot fro ma '.txt' file that is saved
    with the function above. Plot is saved in the
    'results/sensor_population' folder, which is created in the case it
    does not exist.

    Parameters
    ----------
    path : str
        Path to the '.txt' file with precompiled data.
    daughterboard_number : str
        The LinoSPAD2 daughterboard number.
    motherboard_number : str
        The LinoSPAD2 motherboard number.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    app_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    color : str, optional
        Color for the plot. The default is 'salmon'.

    Raises
    ------
    IndexError
        _description_
    """
    os.chdir(path)

    try:
        file = glob.glob("*.txt*")[0]
    except IndexError:
        raise IndexError(".txt file not found - check the folder")

    file_name = file[:-4]

    data = np.genfromtxt(file, delimiter="")

    # Apply mask if requested
    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob(
            "*{}_{}*".format(daughterboard_number, motherboard_number)
        )[0]
        mask = np.genfromtxt(file_mask).astype(int)
        data[mask] = 0
        os.chdir(path_to_back)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(16, 10))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Counts [-]")
    plt.plot(data, "o-", color=color)

    try:
        os.chdir("results/sensor_population")
    except Exception:
        os.makedirs("results/sensor_population")
        os.chdir("results/sensor_population")
    plt.savefig("{}.png".format(file_name))
    os.chdir("..")


def delta_cp_shared(
    path,
    pixels,
    rewrite: bool,
    range_left: int = -10e3,
    range_right: int = 10e3,
    step: int = 1,
    same_y: bool = False,
    color: str = "salmon",
):
    """Collect and plot timestamp differences from a '.csv' file.

    Plots timestamp differences from a '.csv' file as a grid of histograms
    and as a single plot. For plotting from the shared '.csv' files.

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

    # file = glob.glob("*.csv*")[0]
    file = glob.glob("*.feather*")[0]
    csv_file_name = file[:-4]

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
                    file,
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
                    color=chosen_color,
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
                peak_max = None

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
                plt.title(
                    "Pixels {p1},{p2}".format(p1=pixels[q], p2=pixels[w])
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
