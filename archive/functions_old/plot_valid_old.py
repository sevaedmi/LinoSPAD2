"""Module that contains functions cut from the 'functions' as these
are no longer utilized, only for debugging.

Following functions can be found in this module.

    * plot_pixel_hist
    * plot_valid
    * plot_valid_2208

"""

import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up


def plot_pixel_hist(
    path,
    pix,
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
    pix : array-like
        Array of pixels indices.
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

        if fw_ver == "2208":
            data = f_up.unpack_numpy(num, board_number, timestamps)
        elif fw_ver == "2212b":
            # TODO change to faster numpy
            data = f_up.unpack_bin(
                num, board_number, fw_ver="block", timestamps=timestamps
            )

        if pix is None:
            pixels = np.arange(145, 165, 1)
        else:
            pixels = pix
        for i, pixel in enumerate(pixels):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({"font.size": 22})
            # bins = np.arange(0, 4e9, 17.867 * 1e6)  # bin size of 17.867 us
            bins = np.linspace(0, 4e9, 200)
            if fw_ver == "2208":
                plt.hist(data[pixel], bins=bins, color="teal")
            if fw_ver == "2212b":
                plt.hist(data["{}".format(pixel)], bins=bins, color="teal")
            plt.xlabel("Time [ms]")
            plt.ylabel("Counts [-]")
            plt.title("Pixel {}".format(pixel))
            try:
                os.chdir("results/single pixel histograms")
            except Exception:
                os.makedirs("results/single pixel histograms")
                os.chdir("results/single pixel histograms")
            plt.savefig(
                "{file}, pixel {pixel}.png".format(file=num, pixel=pixel)
            )
            os.chdir("../..")


def plot_valid(
    path,
    board_number,
    timestamps: int = 512,
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    app_mask: bool = True,
):
    """Plot number of timestamps in each pixel for single datafile.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    board_number : str
        The LinoSPAD2 board number. Required for choosing the correct
        calibration data.
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
    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    valid_per_pixel = np.zeros(256)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    for i, num in enumerate(DATA_FILES):
        print("> > > Plotting timestamps, Working on {} < < <\n".format(num))

        data_matrix = f_up.unpack_numpy(num, board_number, timestamps)

        for j in range(len(data_matrix)):
            valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

        # Apply mask if requested
        if app_mask is True:
            path_to_back = os.getcwd()
            path_to_mask = (
                os.path.realpath(__file__) + "/../.." + "/params/masks"
            )
            os.chdir(path_to_mask)
            file_mask = glob.glob("*{}*".format(board_number))[0]
            mask = np.genfromtxt(file_mask).astype(int)
            valid_per_pixel[mask] = 0
            os.chdir(path_to_back)

        peak = np.max(valid_per_pixel)

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
        plt.figure(figsize=(16, 10))
        plt.rcParams.update({"font.size": 20})
        plt.title("{file}\n Peak is {peak}".format(file=num, peak=peak))
        plt.xlabel("Pixel [-]")
        plt.ylabel("Timestamps [-]")
        if scale == "log":
            plt.yscale("log")
        plt.plot(valid_per_pixel, style, color=chosen_color)

        try:
            os.chdir("results")
        except Exception:
            os.makedirs("results")
            os.chdir("results")
        plt.savefig("{}.png".format(num))
        plt.pause(0.1)
        if show_fig is False:
            plt.close("all")
        os.chdir("..")


def plot_valid_2208(
    path,
    board_number,
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
    the same folder where datafiles are.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    board_number : str
        The LinoSPAD2 board number. Required for choosing the correct
        calibration data.
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
        Switch for applying the mask on warm/hot pixels. Default is
        True.

    Returns
    -------
    None.

    """
    os.chdir(path)

    files_all = glob.glob("*.dat*")

    plot_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    valid_per_pixel = np.zeros(256)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    print(
        "> > > Plotting valid timestamps, Working in {} < < <\n".format(path)
    )
    for i in tqdm(range(len(files_all)), desc="Collecting data"):
        file = files_all[i]
        data = f_up.unpack_numpy(
            file, board_number, timestamps, app_mask=app_mask
        )

        for j in range(len(data)):
            valid_per_pixel[j] = valid_per_pixel[j] + len(
                # np.where(data["{}".format(j)] > 0)[0]
                np.where(data[j] > 0)[0]
            )

    # Apply mask if requested
    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
        os.chdir(path_to_mask)
        file_mask = glob.glob("*{}*".format(board_number))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        valid_per_pixel[mask] = 0
        os.chdir(path_to_back)

    peak = int(np.max(valid_per_pixel))

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

    print("\n> > > Preparing the plot < < <\n")

    plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 20})
    plt.title("Peak is {peak}".format(peak=peak))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Timestamps [-]")
    if scale == "log":
        plt.yscale("log")
    plt.plot(valid_per_pixel, style, color=chosen_color)

    try:
        os.chdir("results")
    except Exception:
        os.makedirs("results")
        os.chdir("results")
    plt.savefig("{name}.png".format(name=plot_name))

    plt.pause(0.1)
    if show_fig is False:
        plt.close("all")
    os.chdir("..")
