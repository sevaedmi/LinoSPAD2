"""Module with scripts for plotting the LinoSPAD2 output.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 data output.

This file can also be imported as a module and contains the following
functions:

    * plot_pixel_hist - plots a histogram of timestamps for a single
    pixel. The function can be used mainly for checking the
    homogenity of the LinoSPAD2 output.

    * plot_sen_pop - plots the sensor population as a number of valid
    timestamps in each pixel. Works with the firmware version 2212 (both
    block and skip). Analyzes all datafiles in the given folder.

"""

import glob
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from LinoSPAD2.functions import unpack as f_up


def plot_pixel_hist(
    path,
    pixels,
    fw_ver: str,
    board_number: str,
    timestamps: int = 512,
    show_fig: bool = False,
):
    """Plot a histogram for each pixel in the given range.

    Used mainly for checking the homogenity of the LinoSPAD2 output
    (mainly clock and acquisition window size settings).

    Parameters
    ----------
    path : str
        Path to data file.
    pixels : array-like, list
        Array of pixels indices.
    fw_ver : str
        LinoSPAD2 firmware version.
    board_number : str
        LinoSPAD2 daugtherboard number.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The
        default is 512.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string")

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

        data = f_up.unpack_bin(num, board_number, timestamps=timestamps)

        if pixels is None:
            pixels = np.arange(145, 165, 1)

        for i in range(len(pixels)):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({"font.size": 22})
            # bins = np.arange(0, 4e9, 17.867 * 1e6)  # bin size of 17.867 us
            bins = np.linspace(0, 4e9, 200)
            if fw_ver == "2212s":
                pix_coor = np.arange(256).reshape(4, 64).T
            elif fw_ver == "2212b":
                pix_coor = np.arange(256).reshape(64, 4)
            else:
                print("\nFirmware version is not recognized, exiting.")
                sys.exit()
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            data_to_plot = data[tdc].T[1][ind[ind1]]

            plt.hist(data_to_plot, bins=bins, color="teal")
            plt.xlabel("Time [ms]")
            plt.ylabel("Counts [-]")
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
    board_number: str,
    fw_ver: str,
    timestamps: int = 512,
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    app_mask: bool = True,
):
    """Plot number of timestamps in each pixel for all datafiles.

    Plot sensor population as number of timestamps vs pixel number.
    Analyzes all datafiles in the given folder. Output figure is saved
    in the "results" folder, which is created if it does not exist, in
    the same folder where datafiles are. Works with the firmware version
    '2212'.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    board_number : str
        The LinoSPAD2 board number. Required for choosing the correct
        calibration data.
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

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212b' or '2212s'")
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, 'NL11' or 'A5'")
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    files = glob.glob("*.dat*")

    plot_name = files[0][:-4] + "-" + files[-1][:-4]

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

        data = f_up.unpack_bin(files[i], board_number, timestamps)
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
        file_mask = glob.glob("*{}*".format(board_number))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        valid_per_pixel[mask] = 0
        os.chdir(path_to_back)

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(16, 10))
    plt.plot(valid_per_pixel, "-o", color=chosen_color)
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
    path, board_number: str, timestamps: int = 512, show_fig: bool = False
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
    board_number : str
        LinoSPAD2 daugtherboard number. Either "A5" or "NL11" is
        accepted.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The
        default is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

    Raises
    ------
    TypeError
        Raised when 'board_number' is not string.
    ValueError
        Raised when the number of datafiles of SPDC data is different
        from the number of datafiles of background data.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError(
            "'board_number' should be string, either 'NL11' or 'A5'"
        )

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
            files[i], board_number="A5", timestamps=timestamps
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
            files_bckg[i], board_number="A5", timestamps=timestamps
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
    file_mask = glob.glob("*{}*".format(board_number))[0]
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
