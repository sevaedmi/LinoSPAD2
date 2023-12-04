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
from scipy import signal as sg


def fit_wg_double(
    path,
    pix_pair: list,
    thrs: float = 1.2,
    window: float = 5e3,
    step: int = 1,
    color_d: str = "salmon",
    color_f: str = "teal",
    title_on: bool = True,
):
    """Fit with Gaussian function and plot it.
    #TODO

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
    # Use te given window for trimming the data for fitting
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

    peak_pos = sg.find_peaks(n, height=np.median(n) * thrs)[0]
    # print(peak_pos, n[peak_pos])

    # try:
    #     n_argmax = np.argmax(n)
    #     cp_pos = (bins[n_argmax] + bins[n_argmax + 1]) / 2
    # except ValueError:
    #     print("Couldn't find position of histogram max")

    data_to_fit1 = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[peak_pos[0]] - window / 2)
    )
    data_to_fit1 = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[peak_pos[0]] + window / 2)
    )

    data_to_fit2 = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[peak_pos[1]] - window / 2)
    )
    data_to_fit2 = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[peak_pos[1]] + window / 2)
    )

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    n1, b1 = np.histogram(data_to_fit1, bins)
    n2, b2 = np.histogram(data_to_fit2, bins)

    b11 = (b1 - 17.857 * step / 2)[1:]
    b22 = (b2 - 17.857 * step / 2)[1:]

    sigma = 150

    av_bkg = np.average(n)

    par1, pcov1 = curve_fit(
        gauss, b11, n, p0=[max(n), b[peak_pos[0]], sigma, av_bkg]
    )
    par2, pcov2 = curve_fit(
        gauss, b22, n, p0=[max(n), b[peak_pos[1]], sigma, av_bkg]
    )

    # interpolate for smoother fit plot
    to_fit_b1 = np.linspace(np.min(b11), np.max(b11), len(b11) * 100)
    to_fit_n1 = gauss(to_fit_b1, par1[0], par1[1], par1[2], par1[3])

    to_fit_b2 = np.linspace(np.min(b22), np.max(b22), len(b22) * 100)
    to_fit_n2 = gauss(to_fit_b2, par2[0], par2[1], par2[2], par2[3])

    perr1 = np.sqrt(np.diag(pcov1))
    vis_er1 = par1[0] / par1[3] ** 2 * 100 * perr1[-1]
    perr2 = np.sqrt(np.diag(pcov2))
    vis_er2 = par2[0] / par2[3] ** 2 * 100 * perr2[-1]

    plt.figure(figsize=(16, 10))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b1[1:],
        n,
        color=color_d,
        label="data",
    )
    plt.plot(
        # b,
        # fit_plot,
        to_fit_b1,
        to_fit_n1,
        "-",
        color=color_f,
        label="CT\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "vis={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            # "\u03C3_s={p3} ps".format(
            p1=format(par1[2], ".1f"),
            p2=format(par1[1], ".1f"),
            pe1=format(perr1[2], ".1f"),
            pe2=format(perr1[1], ".1f"),
            bkg=format(par1[3], ".1f"),
            bkg_er=format(perr1[3], ".1f"),
            vis=format(par1[0] / par1[3] * 100, ".1f"),
            vis_er=format(vis_er1, ".1f")
            # p3=format(st_dev, ".2f"),
        ),
    )

    plt.plot(
        # b,
        # fit_plot,
        to_fit_b2,
        to_fit_n2,
        "-",
        color="navy",
        label="HBT\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "vis={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            # "\u03C3_s={p3} ps".format(
            p1=format(par2[2], ".1f"),
            p2=format(par2[1], ".1f"),
            pe1=format(perr2[2], ".1f"),
            pe2=format(perr2[1], ".1f"),
            bkg=format(par2[3], ".1f"),
            bkg_er=format(perr2[3], ".1f"),
            vis=format(par2[0] / par2[3] * 100, ".1f"),
            vis_er=format(vis_er2, ".1f")
            # p3=format(st_dev, ".2f"),
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


def fit_wg_all(
    path,
    pix_pair: list,
    thrs: float = 1.2,
    window: float = 5e3,
    step: int = 1,
    color_d: str = "salmon",
    color_f: str = "teal",
    title_on: bool = True,
):
    """Fit with Gaussian function and plot it.
    #TODO

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
    # Use te given window for trimming the data for fitting
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -window))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > window))

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    n, b = np.histogram(data_to_plot, bins)

    peak_pos = sg.find_peaks(n, height=np.median(n) * thrs)[0]

    plt.figure(figsize=(16, 10))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b[1:],
        n,
        color=color_d,
        label="data",
    )

    color_f = ["teal", "navy", "limegreen", "orchid", "paleturquoise"]

    for k, peak_ind in enumerate(peak_pos):
        data_to_fit1 = np.delete(
            data_to_plot, np.argwhere(data_to_plot < b[peak_ind] - window / 2)
        )
        data_to_fit1 = np.delete(
            data_to_plot, np.argwhere(data_to_plot > b[peak_ind] + window / 2)
        )

        # bins must be in units of 17.857 ps
        bins = np.arange(
            np.min(data_to_plot), np.max(data_to_plot), 17.857 * step
        )

        n1, b1 = np.histogram(data_to_fit1, bins)

        b11 = (b1 - 17.857 * step / 2)[1:]

        sigma = 150

        av_bkg = np.average(n)

        par1, pcov1 = curve_fit(
            gauss, b11, n, p0=[max(n), b[peak_pos[0]], sigma, av_bkg]
        )

        # interpolate for smoother fit plot
        to_fit_b1 = np.linspace(np.min(b11), np.max(b11), len(b11) * 100)
        to_fit_n1 = gauss(to_fit_b1, par1[0], par1[1], par1[2], par1[3])

        perr1 = np.sqrt(np.diag(pcov1))
        vis_er1 = par1[0] / par1[3] ** 2 * 100 * perr1[-1]

        plt.plot(
            # b,
            # fit_plot,
            to_fit_b1,
            to_fit_n1,
            "-",
            color=color_f[k],
            label="CT\n"
            "\u03C3={p1}\u00B1{pe1} ps\n"
            "\u03BC={p2}\u00B1{pe2} ps\n"
            "vis={vis}\u00B1{vis_er} %\n"
            "bkg={bkg}\u00B1{bkg_er}".format(
                # "\u03C3_s={p3} ps".format(
                p1=format(par1[2], ".1f"),
                p2=format(par1[1], ".1f"),
                pe1=format(perr1[2], ".1f"),
                pe2=format(perr1[1], ".1f"),
                bkg=format(par1[3], ".1f"),
                bkg_er=format(perr1[3], ".1f"),
                vis=format(par1[0] / par1[3] * 100, ".1f"),
                vis_er=format(vis_er1, ".1f")
                # p3=format(st_dev, ".2f"),
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


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\Ne\703\CT_vs_HBT\1_2"
# %matplotlib qt
fit_wg_all(path, pix_pair=[105, 109], window=15e3, step=12)
