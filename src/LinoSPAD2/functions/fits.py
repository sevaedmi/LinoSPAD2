"""Module for fitting timestamp differences with Gaussian function.

This file can also be imported as a module and contains the following
functions:

    * fit_with_gaussian - fit timestamp differences of a pair of pixels
    with a Gaussian function and plot a histogram of timestamp
    differences and the fit in a single figure.

    * fit_with_gaussian_full_sensor - fit timestamp differences of a
    pair of pixels (one from each half of the sensor) with a Gaussian
    function and plot a histogram of timestamp differences and the fit
    in a single figure.

"""

import glob
import os
from typing import List

import numpy as np
import pyarrow.feather as feather
from matplotlib import pyplot as plt

from LinoSPAD2.functions import utils


def fit_with_gaussian(
    path,
    pix_pair: List[int],
    ft_file: str = None,
    window: float = 5e3,
    step: int = 1,
    color_data: str = "salmon",
    color_fit: str = "teal",
    title_on: bool = True,
):
    # TODO
    """Fit with Gaussian function and plot it.

    Fits timestamp differences of a pair of pixels with Gaussian
    function and plots it next to the histogram of the differences.
    Timestamp differences are collected from a '.feather' file with
    those if such exists.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pix_pair : list
        Two pixel numbers for which fit is done.
    window : float, optional
        Time range in which timestamp differences are fitted. The
        default is 5e3.
    step : int, optional
        Bins of delta t histogram should be in units of 17.857 (average
        LinoSPAD2 TDC bin width), this parameter helps with changing the
        bin size while maintaining that rule. Default is 1.
    color_data : str, optional
        For changing the color of the data. The default is "salmon".
    color_fit : str, optional
        For changing the color of the fit. The default is "teal".
    title_on : bool, optional
        Switch for turning on/off the title of the plot, the title
        shows the pixels for which the fit is done. The default is True.

    Raises
    ------
    FileNotFoundError
        Raised when no '.dat' data files are found.
    FileNotFoundError
        Raised when no '.feather' file with timestamp differences is found.
    ValueError
        Raised when no data for the requested pair of pixels was found
        in the '.feather' file.

    Returns
    -------
    None.

    """
    plt.ion()

    os.chdir(path)

    if ft_file is not None:
        file_name = ft_file.split(".")[0]
        feather_file_name = ft_file
    else:
        files = sorted(glob.glob("*.dat*"))
        file_name = files[0][:-4] + "-" + files[-1][:-4]

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            raise ("\nFile with data not found")

        feather_file_name = glob.glob("*{}.feather*".format(file_name))[0]

    # Version using csv files instead of feather ones.
    # Left for debugging.
    # csv_file_name = glob.glob("*{}*".format(file_name))[0]
    # if csv_file_name == []:
    #     raise FileNotFoundError("\nFile with data not found")

    # data = pd.read_csv(
    #     "{}".format(csv_file_name),
    #     usecols=["{},{}".format(pix_pair[0], pix_pair[1])],
    # )
    # try:
    #     data_to_plot = data["{},{}".format(pix_pair[0], pix_pair[1])]
    # except KeyError:
    #     print("\nThe requested pixel pair is not found")
    # del data

    if not feather_file_name:
        raise FileNotFoundError("\nFile with data not found")

    data_to_plot = feather.read_feather(
        "{}".format(feather_file_name),
        columns=["{},{}".format(pix_pair[0], pix_pair[1])],
    ).dropna()

    # Check if there any finite values
    if not np.any(~np.isnan(data_to_plot)):
        raise ValueError("\nNo data for the requested pixel pair available")

    data_to_plot = data_to_plot.dropna()
    data_to_plot = np.array(data_to_plot)
    # Use the given window for trimming the data for fitting
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -window))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > window))

    os.chdir(path)
    plt.rcParams.update({"font.size": 22})

    # bins must be in units of 17.857 ps
    bins = np.arange(
        np.min(data_to_plot), np.max(data_to_plot), 2.5 / 140 * 1e3 * step
    )

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    n, b = np.histogram(data_to_plot, bins)

    try:
        n_argmax = np.argmax(n)
    except ValueError:
        print("Couldn't find position of histogram max")

    data_to_plot = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[n_argmax] - window / 2)
    )
    data_to_plot = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[n_argmax] + window / 2)
    )

    # bins must be in units of 17.857 ps
    bins = np.arange(
        np.min(data_to_plot), np.max(data_to_plot), 2.5 / 140 * 1e3 * step
    )

    n, b = np.histogram(data_to_plot, bins)

    bin_centers = (b - 2.5 / 140 * 1e3 * step / 2)[1:]

    par, pcov = utils.fit_gaussian(bin_centers, n)

    # interpolate for smoother fit plot
    to_fit_b = np.linspace(
        np.min(bin_centers), np.max(bin_centers), len(bin_centers) * 100
    )
    to_fit_n = utils.gaussian(to_fit_b, par[0], par[1], par[2], par[3])

    perr = np.sqrt(np.diag(pcov))
    vis = par[0] / par[3] * 100
    vis_er = utils.error_propagation_division(par[0], perr[0], par[3], perr[3])
    # Contrast error in %
    vis_er = vis_er / (vis / 100) * 100

    plt.figure(figsize=(16, 10))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b[1:],
        n,
        color=color_data,
        label="data",
    )
    plt.plot(
        to_fit_b,
        to_fit_n,
        "-",
        color=color_fit,
        label="fit\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "C={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            p1=format(par[2], ".1f"),
            p2=format(par[1], ".1f"),
            pe1=format(perr[2], ".1f"),
            pe2=format(perr[1], ".1f"),
            bkg=format(par[3], ".1f"),
            bkg_er=format(perr[3], ".1f"),
            vis=format(vis, ".1f"),
            vis_er=format(vis_er, ".1f"),
        ),
    )
    plt.legend(loc="best")
    if title_on is True:
        plt.title(
            "Gaussian fit of delta t histogram, pixels {}, {}".format(
                pix_pair[0], pix_pair[1]
            )
        )

    try:
        os.chdir("results/fits")
    except Exception:
        os.makedirs("results/fits")
        os.chdir("results/fits")

    plt.savefig(
        "{file}_pixels_"
        "{pix1},{pix2}_fit.png".format(
            file=file_name, pix1=pix_pair[0], pix2=pix_pair[1]
        )
    )

    plt.pause(0.1)
    os.chdir("../..")


def fit_with_gaussian_full_sensor(
    path,
    pix_pair: List[int],
    window: float = 5e3,
    step: int = 1,
    color_data: str = "salmon",
    color_fit: str = "teal",
    title_on: bool = True,
):
    """Fit with Gaussian function and plot it.

    Fits timestamp differences of a pair of pixels with Gaussian
    function and plots it next to the histogram of the differences.
    Timestamp differences are collected from a '.csv' file with those if
    such exists.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pix_pair : list
        Two pixel numbers for which fit is done.
    window : float, optional
        Time range in which timestamp differences are fitted. The
        default is 5e3.
    step : int, optional
        Bins of delta t histogram should be in units of 17.857 (average
        LinoSPAD2 TDC bin width), this parameter helps with changing the
        bin size while maintaining that rule. Default is 1.
    color_data : str, optional
        For changing the color of the data. The default is "salmon".
    color_fit : str, optional
        For changing the color of the fit. The default is "teal".
    title_on : bool, optional
        Switch for turning on/off the title of the plot, the title
        shows the pixels for which the fit is done. The default is True.

    Raises
    ------
    FileNotFoundError
        Raised when no '.dat' data files are found.
    FileNotFoundError
        Raised when no '.feather' file with timestamp differences is found.
    ValueError
        Raised when no data for the requested pair of pixels was found
        in the '.feather' file.

    Returns
    -------
    None.

    """
    os.chdir(path)

    # Assuming the current directory contains the folders
    folders = glob.glob("*#*")

    # Ensure at least two folders are found
    if len(folders) < 2:
        raise ValueError(
            "At least two folders with the specified pattern are required."
        )
    # Depending on the order of the folders, the ".feather" file naming
    # could be different. The following code collects the names for
    # the two possible naming results

    # Collect ".dat" files' names in the first folder
    os.chdir(folders[0])
    files_all = sorted(glob.glob("*.dat*"))
    feather_file_name1 = files_all[0][:-4] + "-"
    feather_file_name2 = "-" + files_all[-1][:-4]

    # Collect ".dat" files' names in the second folder
    os.chdir("../{}".format(folders[1]))
    files_all = sorted(glob.glob("*.dat"))
    feather_file_name1 += files_all[-1][:-4]
    feather_file_name2 = files_all[0][:-4] + feather_file_name2

    os.chdir("..")

    feather_file_path1 = "delta_ts_data/{}.feather".format(feather_file_name1)
    feather_file_path2 = "delta_ts_data/{}.feather".format(feather_file_name2)
    feather_file, feather_file_name = (
        (feather_file_path1, feather_file_name1)
        if os.path.isfile(feather_file_path1)
        else (feather_file_path2, feather_file_name2)
    )

    if not os.path.isfile(feather_file):
        raise FileNotFoundError(
            "'.feather' file with timestamps differences was not found"
        )

    # feather_file_name = glob.glob("*{}.feather*".format(file_name))[0]
    # if not feather_file_name:
    #     raise FileNotFoundError("\nFile with data not found")

    data_to_plot = feather.read_feather(
        "{}".format(feather_file),
        columns=["{},{}".format(pix_pair[0], pix_pair[1])],
    ).dropna()

    # Check if there any finite values
    if not np.any(~np.isnan(data_to_plot)):
        raise ValueError("\nNo data for the requested pixel pair available")

    data_to_plot = data_to_plot.dropna()
    data_to_plot = np.array(data_to_plot)
    # Use the given window for trimming the data for fitting
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -window))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > window))

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    n, b = np.histogram(data_to_plot, bins)

    #
    try:
        n_argmax = np.argmax(n)
        cp_pos = (bins[n_argmax] + bins[n_argmax + 1]) / 2
    except ValueError:
        print("Couldn't find position of histogram max")

    data_to_plot = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[n_argmax] - window / 2)
    )
    data_to_plot = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[n_argmax] + window / 2)
    )

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    n, b = np.histogram(data_to_plot, bins)

    bin_centers = (b - 17.857 * step / 2)[1:]

    par, pcov = utils.fit_gaussian(bin_centers, n)

    # interpolate for smoother fit plot
    to_fit_b = np.linspace(
        np.min(bin_centers), np.max(bin_centers), len(bin_centers) * 100
    )
    to_fit_n = utils.gaussian(to_fit_b, par[0], par[1], par[2], par[3])

    perr = np.sqrt(np.diag(pcov))
    vis = par[0] / par[3] * 100
    vis_er = utils.error_propagation_division(par[0], perr[0], par[3], perr[3])
    # Contrast error in %
    vis_er = vis_er / (vis / 100) * 100

    plt.figure(figsize=(16, 10))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b[1:],
        n,
        color=color_data,
        label="data",
    )
    plt.plot(
        to_fit_b,
        to_fit_n,
        "-",
        color=color_fit,
        label="fit\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "C={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            p1=format(par[2], ".1f"),
            p2=format(par[1], ".1f"),
            pe1=format(perr[2], ".1f"),
            pe2=format(perr[1], ".1f"),
            bkg=format(par[3], ".1f"),
            bkg_er=format(perr[3], ".1f"),
            vis=format(vis, ".1f"),
            vis_er=format(vis_er, ".1f"),
        ),
    )
    plt.legend(loc="best")
    if title_on is True:
        plt.title(
            "Gaussian fit of delta t histogram, pixels {}, {}".format(
                pix_pair[0], pix_pair[1]
            )
        )

    try:
        os.chdir(os.path.join(path, r"results/fits"))
    except Exception:
        os.makedirs(os.path.join(path, r"results/fits"))
        os.chdir(os.path.join(path, r"results/fits"))

    plt.savefig(
        "{file}_pixels_"
        "{pix1},{pix2}_fit.png".format(
            file=feather_file_name, pix1=pix_pair[0], pix2=pix_pair[1]
        )
    )

    plt.pause(0.1)
    os.chdir("../..")
