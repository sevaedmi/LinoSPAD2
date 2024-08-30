"""Module for fitting timestamp differences with Gaussian function.

This file can also be imported as a module and contains the following
functions:

    * fit_with_gaussian - fit timestamp differences of a pair of pixels
    with a Gaussian function and plot a histogram of timestamp
    differences and the fit in a single figure.
    
    * fit_wg_all - find all peaks above the given threshold and fit
    them individually with a Gaussian function.

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
from scipy import signal as sg

from LinoSPAD2.functions import utils


def fit_with_gaussian(
    path: str,
    pix_pair: List[int],
    ft_file: str = None,
    window: float = 5e3,
    step: int = 1,
    color_data: str = "salmon",
    color_fit: str = "teal",
    title_on: bool = True,
    correct_pix_address: bool = False,
    return_fit_params: bool = False,
):
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
    correct_pix_address : bool, optional
        Correct pixel address for the FPGA board on side 23. The
        default is False.
    return_fit_params : bool, optional
        Switch for returning the fit parameters for further analysis.
        The default is False.

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

    if not feather_file_name:
        raise FileNotFoundError("\nFile with data not found")

    # Save to use in the title
    pixels_title = np.copy(pix_pair)

    if correct_pix_address:
        for i, pixel in enumerate(pix_pair):
            if pixel > 127:
                pix_pair[i] = 255 - pix_pair[i]
            else:
                pix_pair[i] = pix_pair[i] + 128

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
    plt.rcParams.update({"font.size": 25})

    # Bins must be in units of 17.857 ps (2500/140)
    bins = np.arange(
        np.min(data_to_plot), np.max(data_to_plot), 2.5 / 140 * 1e3 * step * 2
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

    # Bins must be in units of 17.857 ps (2500/140)
    bins = np.arange(
        np.min(data_to_plot), np.max(data_to_plot), 2.5 / 140 * 1e3 * step
    )

    n, b = np.histogram(data_to_plot, bins)

    bin_centers = (b - 2.5 / 140 * 1e3 * step / 2)[1:]

    par, pcov = utils.fit_gaussian(bin_centers, n)

    # Interpolate for smoother fit plot
    to_fit_b = np.linspace(
        np.min(bin_centers), np.max(bin_centers), len(bin_centers) * 100
    )
    to_fit_n = utils.gaussian(to_fit_b, par[0], par[1], par[2], par[3])

    perr = np.sqrt(np.diag(pcov))
    contrast = par[0] / par[3] * 100
    vis_er = utils.error_propagation_division(par[0], perr[0], par[3], perr[3])

    # Contrast error in %
    vis_er = vis_er / (contrast / 100) * 100

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
        "C={contrast}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            p1=format(par[2], ".1f"),
            p2=format(par[1], ".1f"),
            pe1=format(perr[2], ".1f"),
            pe2=format(perr[1], ".1f"),
            bkg=format(par[3], ".1f"),
            bkg_er=format(perr[3], ".1f"),
            contrast=format(contrast, ".1f"),
            vis_er=format(vis_er, ".1f"),
        ),
    )
    plt.legend(loc="best")
    if title_on is True:
        plt.title(
            "Gaussian fit of delta t histogram, pixels {}, {}".format(
                pixels_title[0], pixels_title[1]
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

    return (par, perr) if return_fit_params else None


def fit_wg_all(
    path: str,
    pix_pair: List[int],
    threshold: float = 1.2,
    window: float = 5e3,
    step: int = 1,
    color_d: str = "salmon",
    color_f: str = "teal",
    title_on: bool = True,
    correct_pix_address: bool = False,
    return_fit_params: bool = False,
):
    """Find all peaks above threshold and fit each with Gaussian.

    Finds all peaks above the given threshold (uses the 'threshold'
    parameter as a multiplier of the median) and fits each with a
    Gaussian function.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pix_pair : list
        Two pixel numbers for which the fit is done.
    threshold : float, optional
        Multiplier for the median of the histogram counts. Used for
        setting a threshold for the peak discovery in the histogram.
    window : float, optional
        Time range in which timestamp differences are fitted. The
        default is 5e3.
    step : int, optional
        Bins of delta t histogram should be in units of 17.857 (average
        LinoSPAD2 TDC bin width). Default is 1.
    color_d : str, optional
        Color for the histogram. The default is "salmon".
    color_f : str, optional
        Color for the fit with Gaussian. The default is "teal".
    title_on : bool, optional
        Switch for showing the plot title. The default is "True".
    correct_pix_address : bool, optional
        Correct pixel address for the FPGA board on side 23. The
        default is False.
    return_fit_params : bool, optional
        Switch for returning the fit parameters for further analysis.
        The default is False.

    Raises
    ------
    FileNotFoundError
        Raised when no '.dat' data files are found.
    FileNotFoundError
        Raised when no '.feather' file with timestamp differences is
        found.
    ValueError
        Raised when no data for the requested pair of pixels was found
        in the '.feather' file.

    Returns
    -------
    None.

    """

    os.chdir(path)

    files = glob.glob("*.dat*")
    file_name = files[0][:-4] + "-" + files[-1][:-4]

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        raise ("\nFile with data not found")

    feather_file_name = glob.glob("*{}.feather*".format(file_name))[0]
    if not feather_file_name:
        raise FileNotFoundError("\nFile with data not found")

    # Save to use in the title
    pixels_title = np.copy(pix_pair)

    if correct_pix_address:
        for i, pixel in enumerate(pix_pair):
            if pixel > 127:
                pix_pair[i] = 255 - pix_pair[i]
            else:
                pix_pair[i] = pix_pair[i] + 128

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

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    # Bins must be in units of 17.857 ps (2500/140)
    bins = np.arange(
        np.min(data_to_plot), np.max(data_to_plot), 2500 / 140 * step
    )

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    n, b = np.histogram(data_to_plot, bins)

    bin_c = (b - (b[1] - b[0]) / 2)[1:]

    peak_pos = sg.find_peaks(n, height=np.median(n) * threshold)[0]

    plt.figure(figsize=(12, 8))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        bin_c,
        n,
        color=color_d,
        label="data",
    )

    color_f = [
        "teal",
        "navy",
        "limegreen",
        "orchid",
        "paleturquoise",
        "gold",
        "royalblue",
    ]
    plt.xticks(
        [-20e3, -10e3, 0, 10e3, 20e3],
        [
            f"{-20e3:.0f}",
            f"{-10e3:.0f}",
            f"{0:.0f}",
            f"{10e3:.0f}",
            f"{20e3:.0f}",
        ],
    )
    labels = [
        "Peak 1",
        "Peak 2",
        "Peak 3",
        "Peak 4",
        "Peak 5",
        "Peak 6",
        "Peak 7",
    ]

    for k, peak_ind in enumerate(peak_pos):
        data_to_fit = np.delete(
            data_to_plot, np.argwhere(data_to_plot < bin_c[peak_ind] - 3e3)
        )
        data_to_fit = np.delete(
            data_to_fit, np.argwhere(data_to_fit > bin_c[peak_ind] + 3e3)
        )

        # Bins must be in units of 17.857 ps (2500/140)
        bins = np.arange(
            np.min(data_to_fit), np.max(data_to_fit), 2500 / 140 * step
        )

        counts, bin_edges = np.histogram(data_to_fit, bins)

        bin_centers = (bin_edges - (bin_edges[1] - bin_edges[0]) / 2)[1:]

        par, pcov = utils.fit_gaussian(bin_centers, counts)

        # Interpolate for smoother fit plot
        to_fit_n1 = utils.gaussian(bin_centers, par[0], par[1], par[2], par[3])

        perr = np.sqrt(np.diag(pcov))

        contrast = par[0] / par[-1] * 100

        vis_er = utils.error_propagation_division(
            par[0], perr[0], par[3], perr[3]
        )
        vis_er = vis_er / (contrast / 100) * 100

        plt.plot(
            bin_centers,
            to_fit_n1,
            "-",
            color=color_f[k],
            label=f"{labels[k]}\n"
            "\u03C3={p1}\u00B1{pe1} ps\n"
            "\u03BC={p2}\u00B1{pe2} ps\n"
            "C={contrast}\u00B1{vis_er} %".format(
                labels,
                p1=format(par[2], ".0f"),
                p2=format(par[1], ".0f"),
                pe1=format(perr[2], ".0f"),
                pe2=format(perr[1], ".0f"),
                contrast=format(contrast, ".1f"),
                vis_er=format(vis_er, ".1f"),
            ),
        )
        lower_limit = par[1] - 2 * par[2]
        upper_limit = par[1] + 2 * par[2]

        data_in_interval = data_to_fit[
            (data_to_fit >= lower_limit) & (data_to_fit <= upper_limit)
        ]

        bckg_center_position = par[1] - 7 * par[2]

        bckg_in_2sigma = data_to_fit[
            (data_to_fit > bckg_center_position - 2 * par[2])
            & (data_to_fit < bckg_center_position + 2 * par[2])
        ]

        # Plot the Gaussian fit and the 2-sigma interval
        er1 = np.sqrt(len(data_in_interval))
        er2 = np.sqrt(len(bckg_in_2sigma))
        print(
            f"Population of the peak at {bin_c[peak_ind]} in a"
            f"2sigma interval is: {len(data_in_interval) - len(bckg_in_2sigma)}"
            + "\u00B1"
            + f"{np.sqrt(er1**2+er2**2):.2f}"
        )

    plt.xlim(-window, window)
    plt.legend(loc="best")
    if title_on is True:
        plt.title(
            "Gaussian fit of delta t histogram, pixels {}, {}".format(
                pixels_title[0], pixels_title[1]
            )
        )

    try:
        os.chdir("results/fits")
    except FileNotFoundError as _:
        os.makedirs("results/fits")
        os.chdir("results/fits")

    plt.savefig(
        "{file}_pixels_"
        "{pix1},{pix2}_fit.png".format(
            file=file_name, pix1=pixels_title[0], pix2=pixels_title[1]
        )
    )

    plt.pause(0.1)
    os.chdir("../..")

    return (par, perr) if return_fit_params else None


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

    # Bins must be in units of 17.857 ps (2500/140)
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    n, b = np.histogram(data_to_plot, bins)

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

    # Bins must be in units of 17.857 ps (2500/140)
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    n, b = np.histogram(data_to_plot, bins)

    bin_centers = (b - 17.857 * step / 2)[1:]

    par, pcov = utils.fit_gaussian(bin_centers, n)

    # Interpolate for smoother fit plot
    to_fit_b = np.linspace(
        np.min(bin_centers), np.max(bin_centers), len(bin_centers) * 100
    )
    to_fit_n = utils.gaussian(to_fit_b, par[0], par[1], par[2], par[3])

    perr = np.sqrt(np.diag(pcov))
    contrast = par[0] / par[3] * 100
    vis_er = utils.error_propagation_division(par[0], perr[0], par[3], perr[3])

    # Contrast error in %
    vis_er = vis_er / (contrast / 100) * 100

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
        "C={contrast}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            p1=format(par[2], ".1f"),
            p2=format(par[1], ".1f"),
            pe1=format(perr[2], ".1f"),
            pe2=format(perr[1], ".1f"),
            bkg=format(par[3], ".1f"),
            bkg_er=format(perr[3], ".1f"),
            contrast=format(contrast, ".1f"),
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
    except FileNotFoundError as _:
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
