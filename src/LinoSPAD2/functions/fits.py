"""Module for fitting timestamp differences with Gaussian function.

This file can also be imported as a module and contains the following
functions:

    * fit_wg - fit timestamp differences of a pair of pixels with a
    gaussian function and plot both a histogram of timestamp
    differences and the fit in a single figure.

"""

import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def fit_wg(
    path,
    pix_pair: list,
    window: float = 5e3,
    step: int = 1,
    color_d: str = "salmon",
    color_f: str = "teal",
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
        LinoSPAD2 TDC bin width). Default is 1.

    Raises
    ------
    FileNotFoundError
        Raised when no '.dat' data files are found.
    FileNotFoundError
        Raised when no '.csv' file with timestamp differences is found.
    ValueError
        Raised when no data for the requested pair of pixels was found
        in the '.csv' file.

    Returns
    -------
    None.

    """
    plt.ion()

    def gauss(x, A, x0, sigma, C):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

    os.chdir(path)

    files = glob.glob("*.dat*")
    file_name = files[0][:-4] + "-" + files[-1][:-4]

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        raise ("\nFile with data not found")

    csv_file_name = glob.glob("*{}*".format(file_name))[0]
    if csv_file_name == []:
        raise FileNotFoundError("\nFile with data not found")

    data = pd.read_csv(
        "{}".format(csv_file_name),
        usecols=["{},{}".format(pix_pair[0], pix_pair[1])],
    )
    try:
        data_to_plot = data["{},{}".format(pix_pair[0], pix_pair[1])]
    except KeyError:
        print("\nThe requested pixel pair is not found")
    del data
    # Check if there any finite values
    if not np.any(~np.isnan(data_to_plot)):
        raise ValueError("\nNo data for the requested pixel pair available")

    data_to_plot = data_to_plot.dropna()
    data_to_plot = np.array(data_to_plot)
    # Use window of 40 ns for calculating histogram data
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -window))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > window))

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    # n, b, p = plt.hist(data_to_plot, bins=bins, color="teal")
    # plt.close("all")
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

    # n, b, p = plt.hist(data_to_plot, bins=bins, color="teal")
    # plt.close("all")

    n, b = np.histogram(data_to_plot, bins)

    # b1 = (b - (b[-1] - b[-2]) / 2)[1:]
    b1 = (b - 17.857 * step / 2)[1:]

    sigma = 150

    av_bkg = np.average(n)

    par, pcov = curve_fit(gauss, b1, n, p0=[max(n), cp_pos, sigma, av_bkg])

    # interpolate for smoother fit plot
    to_fit_b = np.linspace(np.min(b1), np.max(b1), len(b1) * 100)
    to_fit_n = gauss(to_fit_b, par[0], par[1], par[2], par[3])

    perr = np.sqrt(np.diag(pcov))
    vis_er = par[0] / par[3] ** 2 * 100 * perr[-1]
    # fit_plot = gauss(to_fit_b, par[0], par[1], par[2], par[3])

    plt.figure(figsize=(16, 10))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b[1:],
        n,
        color=color_d,
        label="data",
    )
    plt.plot(
        # b,
        # fit_plot,
        to_fit_b,
        to_fit_n,
        "-",
        color=color_f,
        label="fit\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "vis={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            # "\u03C3_s={p3} ps".format(
            p1=format(par[2], ".1f"),
            p2=format(par[1], ".1f"),
            pe1=format(perr[2], ".1f"),
            pe2=format(perr[1], ".1f"),
            bkg=format(par[3], ".1f"),
            bkg_er=format(perr[3], ".1f"),
            vis=format(par[0] / par[3] * 100, ".1f"),
            vis_er=format(vis_er, ".1f")
            # p3=format(st_dev, ".2f"),
        ),
    )
    plt.legend(loc="best")

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
