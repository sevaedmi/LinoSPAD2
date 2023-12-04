"""Module with scripts for plotting the LinoSPAD2 sensor population.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 data output.

This file can also be imported as a module and contains the following
functions:

    * plot_pixel_hist - plots a histogram of timestamps for a single
    pixel. The function can be used mainly for checking the
    homogeneity of the LinoSPAD2 output.

    * plot_sen_pop - plots the sensor population as a number of
    timestamps in each pixel. Works with the firmware version 2212 (both
    block and skip). Analyzes all data files in the given folder.

    * plot_spdc - plot sensor population for SPDC data. Data files
    taken with background only (SPDC output is off) should be provided.
    The background is subtracted from the actual data for clearer
    plot of the SPDC signal.

"""

import glob
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

from LinoSPAD2.functions import unpack as f_up


def plot_pixel_hist(
    path,
    pixels,
    db_num: str,
    mb_num: str,
    fw_ver: str,
    timestamps: int = 512,
    show_fig: bool = False,
    inc_offset: bool = True,
    app_calib: bool = True,
    fit_average: bool = False,
    color: str = "teal",
):
    """Plot a histogram for each pixel in the given range.

    Used mainly for checking the homogeneity of the LinoSPAD2 output
    (mainly clock and acquisition window size settings).

    Parameters
    ----------
    path : str
        Path to data file.
    pixels : array-like, list
        Array of pixels indices.
    db_num : str
        LinoSPAD2 daughterboard number.
    mb_num : str
        LinoSPAD2 motherboard (FPGA) number.
    fw_ver : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The
        default is 512.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    inc_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    app_calib : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while inc_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    fit_average : int, optional
        Switch for fitting averages of histogram counts in windows of
        +/-10. The default is False.
    color : str, optional
        Color for the histogram. The default is 'teal'.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be a string")
    if isinstance(db_num, str) is not True:
        raise TypeError("'db_num' should be a string")
    if isinstance(mb_num, str) is not True:
        raise TypeError("'mb_num' should be a string")

    def lin_fit(x, a, b):
        return a * x + b

    if type(pixels) is int:
        pixels = [pixels]

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()
    for i, num in enumerate(DATA_FILES):
        print(
            "> > > Plotting pixel histograms, Working on {} < < <\n".format(
                num
            )
        )

        data = f_up.unpack_bin(
            num, db_num, mb_num, fw_ver, timestamps, inc_offset, app_calib
        )

        bins = np.arange(0, 4e9, 17.867 * 1e6)  # bin size of 17.867 us

        if pixels is None:
            pixels = np.arange(145, 165, 1)

        for i in range(len(pixels)):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({"font.size": 22})
            if fw_ver == "2212s":
                pix_coor = np.arange(256).reshape(4, 64).T
            elif fw_ver == "2212b":
                pix_coor = np.arange(256).reshape(64, 4)
            else:
                print("\nFirmware version is not recognized, exiting.")
                sys.exit()
            tdc, pix = np.argwhere(pix_coor == pixels[i])[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            data_to_plot = data[tdc].T[1][ind[ind1]]

            n, b, p = plt.hist(data_to_plot, bins=bins, color=color)
            if fit_average is True:
                av_win = np.zeros(int(len(n) / 10) + 1)
                av_win_in = np.zeros(int(len(n) / 10) + 1)
                for j in range(len(av_win)):
                    av_win[j] = n[j * 10 : j * 10 + 1]
                    av_win_in[j] = b[j * 10 : j * 10 + 1]

                a = 1
                b = np.average(n)

                par, pcov = curve_fit(lin_fit, av_win_in, av_win, p0=[a, b])

                av_win_fit = lin_fit(av_win_in, par[0], par[1])

            plt.xlabel("Time [ps]")
            plt.ylabel("Counts [-]")
            # plt.plot(av_win_in, av_win, color="black", linewidth=8)
            if fit_average is True:
                plt.gcf()
                plt.plot(av_win_in, av_win_fit, color="black", linewidth=8)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.title("Pixel {}".format(pixels[i]))
            try:
                os.chdir("results/single pixel histograms")
            except Exception:
                os.makedirs("results/single pixel histograms")
                os.chdir("results/single pixel histograms")
            plt.savefig(
                "{file}, pixel {pixel}.png".format(file=num, pixel=pixels[i])
            )
            os.chdir("../..")


def plot_sen_pop(
    path,
    db_num: str,
    mb_num: str,
    fw_ver: str,
    timestamps: int = 512,
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    app_mask: bool = True,
    inc_offset: bool = True,
    app_calib: bool = True,
    color: str = "salmon",
):
    """Plot number of timestamps in each pixel for all datafiles.

    Plot sensor population as number of timestamps vs. pixel number.
    Analyzes all data files in the given folder. The output figure is saved
    in the "results" folder, which is created if it does not exist, in
    the same folder where datafiles are. Works with the firmware version
    '2212'.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    db_num : str
        The LinoSPAD2 daughterboard number. Required for choosing the
        correct calibration data.
    mb_num : str
        The LinoSPAD2 motherboard number.
    fw_ver : str
        LinoSPAD2 firmware version. Versions '2212b' (block) or '2212s'
        (skip) are recognized.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. Default is
        "512".
    scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    app_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    inc_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    app_calib : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while inc_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    color : str, optional
        Color for the plot. The default is 'salmon'.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212b' or '2212s'")
    if isinstance(db_num, str) is not True:
        raise TypeError("'db_num' should be string, 'NL11' or 'A5'")
    if isinstance(mb_num, str) is not True:
        raise TypeError("'mb_num' should be string")
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    files = glob.glob("*.dat*")

    plot_name = files[0][:-4] + "-" + files[-1][:-4]

    valid_per_pixel = np.zeros(256)

    print(
        "\n> > > Collecting data for sensor population plot,"
        "Working in {} < < <\n".format(path)
    )

    for i in tqdm(range(len(files)), desc="Collecting data"):
        if fw_ver == "2212s":
            pix_coor = np.arange(256).reshape(4, 64).T
        elif fw_ver == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized, exiting.")
            sys.exit()

        data = f_up.unpack_bin(
            files[i], db_num, mb_num, fw_ver, timestamps, inc_offset, app_calib
        )
        for i in range(256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            valid_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])

    print("\n> > > Plotting < < <\n")
    # Apply mask if requested
    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob("*{}_{}*".format(db_num, mb_num))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        valid_per_pixel[mask] = 0
        os.chdir(path_to_back)

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(16, 10))
    if scale == "log":
        plt.yscale("log")
    plt.plot(valid_per_pixel, style, color=color)
    plt.xlabel("Pixel number [-]")
    plt.ylabel("Timestamps [-]")

    try:
        os.chdir("results/sensor_population")
    except FileNotFoundError:
        os.makedirs("results/sensor_population")
        os.chdir("results/sensor_population")
    fig.tight_layout()
    plt.savefig("{}.png".format(plot_name))
    print(
        "> > > The plot is saved as '{file}.png' in {path} < < <".format(
            file=plot_name, path=os.getcwd()
        )
    )
    os.chdir("../..")


def plot_spdc(
    path,
    db_num: str,
    mb_num: str,
    fw_ver: str,
    timestamps: int = 512,
    show_fig: bool = False,
):
    """Plot sensor population for SPDC data.

    Plots SPDC data subtracting the background (data with the SPDC
    output turned off). Due to the low sensitivity of the LinoSPAD2
    sensor to light at 810 nm of Thorlabs SPDC output, subtracting
    background is required to show any meaningful signal.

    Parameters
    ----------
    path : str
        Path to data files.
    db_num : str
        LinoSPAD2 daughterboard number. Either "A5" or "NL11" is
        accepted.
    mb_num : str
        LinoSPAD2 motherboard number.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The
        default is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

    Raises
    ------
    TypeError
        Raised when 'db_num' is not a string.
    ValueError
        Raised when the number of data files of SPDC data is different
        from the number of data files of background data.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(db_num, str) is not True:
        raise TypeError("'db_num' should be string, either 'NL11' or 'A5'")
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string")
    if isinstance(mb_num, str) is not True:
        raise TypeError("'db_num' should be string, either 'NL11' or 'A5'")

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    files = glob.glob("*.dat*")

    # background data for subtracting
    path_bckg = path + "/bckg"
    os.chdir(path_bckg)
    files_bckg = glob.glob("*.dat*")
    os.chdir("..")

    if len(files) != len(files_bckg):
        raise ValueError(
            "Number of files with background data is different from the"
            "number of actual data, exiting."
        )

    plot_name = files[0][:-4] + "-" + files[-1][:-4]

    valid_per_pixel = np.zeros(256)
    valid_per_pixel_bckg = np.zeros(256)

    pix_coor = np.arange(256).reshape(64, 4)

    # Collect SPDC data
    for i in tqdm(range(len(files)), desc="Going through datafiles"):
        data_all = f_up.unpack_bin(
            files[i], db_num="A5", fw_ver="2212b", timestamps=timestamps
        )

        for i in np.arange(0, 256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            ind1 = ind[np.where(data_all[tdc].T[1][ind] > 0)[0]]
            valid_per_pixel[i] += len(data_all[tdc].T[1][ind1])

    # Collect background data for subtracting
    os.chdir(path_bckg)

    for i in tqdm(
        range(len(files_bckg)), desc="Going through background datafiles"
    ):
        data_all_bckg = f_up.unpack_bin(
            files_bckg[i],
            db_num="A5",
            mb_num="34",
            fw_ver="2212b",
            timestamps=timestamps,
        )

        # Fot plotting counts
        for i in np.arange(0, 256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data_all_bckg[tdc].T[0] == pix)[0]
            ind1 = ind[np.where(data_all_bckg[tdc].T[1][ind] > 0)[0]]
            valid_per_pixel_bckg[i] += len(data_all_bckg[tdc].T[1][ind1])

    os.chdir("..")

    # Mask the hot/warm pixels
    path_to_back = os.getcwd()
    path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    os.chdir(path_to_mask)
    file_mask = glob.glob("*{}_{}*".format(db_num, mb_num))[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    for i in mask:
        valid_per_pixel[i] = 0
        valid_per_pixel_bckg[i] = 0

    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(16, 10))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Counts [-]")
    plt.title("SPDC data, background subtracted")
    plt.plot(valid_per_pixel - valid_per_pixel_bckg, "o-", color="teal")
    plt.tight_layout()

    try:
        os.chdir("results/sensor_population")
    except Exception:
        os.makedirs("results/sensor_population")
        os.chdir("results/sensor_population")
    plt.savefig("{}_SPDC_counts.png".format(plot_name))
    plt.pause(0.1)
    os.chdir("../..")


def plot_sen_pop_fs(
    path1,
    path2,
    db_num: str,
    mb_num1: str,
    mb_num2: str,
    fw_ver: str,
    timestamps: int = 512,
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    app_mask: bool = True,
    inc_offset: bool = True,
    app_calib: bool = True,
    color: str = "salmon",
):
    # TODO
    """Plot number of timestamps in each pixel for all datafiles.

    Plot sensor population as number of timestamps vs. pixel number.
    Analyzes all data files in the given folder. The output figure is saved
    in the "results" folder, which is created if it does not exist, in
    the same folder where datafiles are. Works with the firmware version
    '2212'.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    db_num : str
        The LinoSPAD2 daughterboard number. Required for choosing the
        correct calibration data.
    mb_num : str
        The LinoSPAD2 motherboard number.
    fw_ver : str
        LinoSPAD2 firmware version. Versions '2212b' (block) or '2212s'
        (skip) are recognized.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. Default is
        "512".
    scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    app_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    inc_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    app_calib : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while inc_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    color : str, optional
        Color for the plot. The default is 'salmon'.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212b' or '2212s'")
    if isinstance(db_num, str) is not True:
        raise TypeError("'db_num' should be string, 'NL11' or 'A5'")
    if isinstance(mb_num1, str) is not True:
        raise TypeError("'mb_num' should be string")
    if isinstance(mb_num2, str) is not True:
        raise TypeError("'mb_num' should be string")
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    valid_per_pixel1 = np.zeros(256)
    valid_per_pixel2 = np.zeros(256)

    # First motherboard / half of the sensor

    os.chdir(path1)

    files = glob.glob("*.dat*")

    plot_name1 = files[0][:-4] + "-"
    print(
        "\n> > > Collecting data for sensor population plot,"
        "Working in {} < < <\n".format(path1)
    )

    for i in tqdm(range(len(files)), desc="Collecting data"):
        if fw_ver == "2212s":
            pix_coor = np.arange(256).reshape(4, 64).T
        elif fw_ver == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized, exiting.")
            sys.exit()

        data = f_up.unpack_bin(
            files[i],
            db_num,
            mb_num1,
            fw_ver,
            timestamps,
            inc_offset,
            app_calib,
        )
        for i in range(256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            valid_per_pixel1[i] += len(data[tdc].T[1][ind[ind1]])

    print("\n> > > Plotting < < <\n")
    # Apply mask if requested
    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob("*{}_{}*".format(db_num, mb_num1))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        valid_per_pixel1[mask] = 0
        os.chdir(path_to_back)

    # Second motherbaord / half of the sensor

    os.chdir(path2)

    files = glob.glob("*.dat*")

    plot_name2 = files[-1][:-4]

    print(
        "\n> > > Collecting data for sensor population plot,"
        "Working in {} < < <\n".format(path2)
    )

    for i in tqdm(range(len(files)), desc="Collecting data"):
        if fw_ver == "2212s":
            pix_coor = np.arange(256).reshape(4, 64).T
        elif fw_ver == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized, exiting.")
            sys.exit()

        data = f_up.unpack_bin(
            files[i],
            db_num,
            mb_num2,
            fw_ver,
            timestamps,
            inc_offset,
            app_calib,
        )
        for i in range(256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            valid_per_pixel2[i] += len(data[tdc].T[1][ind[ind1]])

    print("\n> > > Plotting < < <\n")
    # Apply mask if requested
    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob("*{}_{}*".format(db_num, mb_num2))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        valid_per_pixel2[mask] = 0
        os.chdir(path_to_back)

    valid_per_pixel = np.concatenate(
        [valid_per_pixel1, np.flip(valid_per_pixel2)]
    )

    plot_name = plot_name1 + plot_name2

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(16, 10))
    if scale == "log":
        plt.yscale("log")
    plt.plot(valid_per_pixel, style, color=color)
    plt.xlabel("Pixel number [-]")
    plt.ylabel("Timestamps [-]")

    os.chdir("..")

    try:
        os.chdir("results/sensor_population")
    except FileNotFoundError:
        os.makedirs("results/sensor_population")
        os.chdir("results/sensor_population")
    fig.tight_layout()
    plt.savefig("{}.png".format(plot_name))
    print(
        "> > > The plot is saved as '{file}.png' in {path} < < <".format(
            file=plot_name, path=os.getcwd()
        )
    )
    os.chdir("../..")
