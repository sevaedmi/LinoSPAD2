"""Module for fitting timestamp differences with Gaussian function.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * gauss_fit - function for fitting the peak in the timestamp differences
    with a gaussian function. Uses the calibration data. Analyzes only the
    requested file.

    * fig_gauss_mult - function for fitting the peak in the timestamp
    differences with a gaussian function. Uses the calibration data. Combines
    all data files in the directory for analysis. Memory-friendly version as it
    keeps in memory the data only for the required pixels.

"""

import glob
import os
from statistics import stdev

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def fit_gauss(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    show_fig: bool = False,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """Fit timestamp differences with a gaussian function.

    Function for fitting the peak in the timestamp differences with a
    gaussian function. Uses the calibration data.

    Parameters
    ----------
    path : str
        Path to the data files.
    pix : array-like
        Array of pixels for which the timestamp differences and the gaussian fit
        would be performed.
    board_number: str
        Number of the LinoSPAD2 board.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    range_left: float, optional
        Left limit for timestamp differences.
    range_right: float, optional
        Right limit for timestamp differences.

    Returns
    -------
    None.

    """

    def gauss(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):
        print("\n> > > Fitting with gauss, Working on {} < < <\n".format(filename))
        data = f_up.unpack_numpy(filename, board_number, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))
        for i, num in enumerate(pix):
            data_pix[i] = data[num]

        plt.rcParams.update({"font.size": 20})
        # if len(pix) > 2:
        #     fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
        # else:
        #     fig = plt.figure(figsize=(14, 14))
        print("\n> > > Fitting with gauss < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                # if w <= q:
                #     continue
                data_pair = np.vstack((data_pix[q], data_pix[w]))

                output = cd(
                    data_pair,
                    timestamps=timestamps,
                    range_left=range_left,
                    range_right=range_right,
                )

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"
                if show_fig is True:
                    plt.ion()
                else:
                    plt.ioff()
                try:
                    bins = np.arange(np.min(output), np.max(output), 17.857 * 2)
                except Exception:
                    continue
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n, b, p = plt.hist(output, bins=bins, color=chosen_color)
                plt.close("all")

                try:
                    n_max = np.argmax(n)
                    arg_max = (bins[n_max] + bins[n_max + 1]) / 2
                except Exception:
                    arg_max = None
                    pass
                sigma = 100

                par, covariance = curve_fit(
                    gauss, b[:-1], n, p0=[max(n), arg_max, sigma]
                )
                fit_plot = gauss(b, par[0], par[1], par[2])

                plt.figure(figsize=(16, 10))
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                plt.plot(b[:-1], n, "o", color=chosen_color, label="data")
                plt.plot(
                    b,
                    fit_plot,
                    "-",
                    color="cadetblue",
                    # label="fit\n" "\u03C3={}".format(par[-1]),
                    label="fit\n\u03BC={p1} ps\n"
                    "\u03C3={p2} ps".format(
                        p1=format(par[1], ".2f"), p2=format(par[-1], ".2f")
                    ),
                )
                plt.legend(loc="best")

                try:
                    os.chdir("results/gauss_fit")
                except Exception:
                    os.makedirs("results/gauss_fit")
                    os.chdir("results/gauss_fit")
                plt.savefig(
                    "{file}_pixels"
                    "{pix1},{pix2}_fit.png".format(
                        file=filename, pix1=pix[q], pix2=pix[w]
                    )
                )
                plt.pause(0.1)
                os.chdir("../..")


def fit_gauss_mult(
    path,
    pix,
    board_number: str,
    timestamps: int = 512,
    show_fig: bool = False,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """Fit timestamp differences with a gaussian function, analyze all datafiles.

    Function for fitting the peak in the timestamp differences with a
    gaussian function. Uses the calibration data. Combines all ".dat"
    files in the folder or analyses only the last file created.
    Memory-friendly version as it unpacks only the requested pixels
    (in the case of multiple files).

    Parameters
    ----------
    path : str
        Path to the data files.
    pix : array-like
        Array of pixels for which the timestamp differences and the gaussian fit
        would be performed.
    board_number: str
        Number of the LinoSPAD2 board.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    range_left: float, optional
        Left limit for timestamp differences.
    range_right: float, optional
        Right limit for timestamp differences.

    Returns
    -------
    None.

    """

    def gauss(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    print("\n> > > Fitting with gauss, Working in {} < < <\n".format(path))
    data_pix, plot_name = f_up.unpack_mult_cut(path, pix, board_number, timestamps)

    plt.rcParams.update({"font.size": 20})
    print("\n> > > Fitting with gauss < < <\n")
    for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
        for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
            if w <= q:
                continue
            data_pair = np.vstack((data_pix[q], data_pix[w]))

            output = cd(
                data_pair,
                timestamps=timestamps,
                range_left=range_left,
                range_right=range_right,
            )

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

            if show_fig is True:
                plt.ion()
            else:
                plt.ioff()

            try:
                bins = np.linspace(np.min(output), np.max(output), 100)
            except Exception:
                continue

            plt.xlabel("\u0394t [ps]")
            plt.ylabel("Timestamps [-]")
            n, b, p = plt.hist(output, bins=bins, color=chosen_color)
            plt.close("all")

            try:
                n_max = np.argmax(n)
                arg_max = (bins[n_max] + bins[n_max + 1]) / 2
            except Exception:
                arg_max = None
                pass
            sigma = 100

            try:
                par, covariance = curve_fit(
                    gauss, b[:-1], n, p0=[max(n), arg_max, sigma]
                )
                fit_plot = gauss(b, par[0], par[1], par[2])
            except RuntimeError:
                continue

            plt.figure(figsize=(16, 10))
            plt.xlabel("\u0394t [ps]")
            plt.ylabel("Timestamps [-]")
            plt.plot(b[:-1], n, "o", color=chosen_color, label="data")
            plt.plot(
                b,
                fit_plot,
                "-",
                color="cadetblue",
                # label="fit\n" "\u03C3={}".format(par[-1]),
                label="fit\n\u03BC={p1} ps\n"
                "\u03C3={p2} ps".format(
                    p1=format(par[1], ".2f"), p2=format(par[-1], ".2f")
                ),
            )
            plt.legend(loc="best")

            try:
                os.chdir("results/gauss_fit")
            except Exception:
                os.makedirs("results/gauss_fit")
                os.chdir("results/gauss_fit")

            plt.savefig(
                "{file}_pixels"
                "{pix1},{pix2}_fit.png".format(file=plot_name, pix1=pix[q], pix2=pix[w])
            )

            plt.pause(0.1)
            os.chdir("../..")


def fit_wg(path, pix_pair, window: float = 5e3):
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
        "{}".format(csv_file_name), usecols=["{},{}".format(pix_pair[0], pix_pair[1])]
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
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -20e3))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > 20e3))

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    bins = np.linspace(np.min(data_to_plot), np.max(data_to_plot), 100)

    n, b, p = plt.hist(data_to_plot, bins=bins, color="teal")
    plt.close("all")

    try:
        n_argmax = np.argmax(n)
        cp_pos = (bins[n_argmax] + bins[n_argmax + 1]) / 2
    except ValueError:
        print("Couldn't find position of histogram max")

    data_to_plot = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[n_argmax] - window)
    )
    data_to_plot = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[n_argmax] + window)
    )

    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 100)

    n, b, p = plt.hist(data_to_plot, bins=bins, color="teal")
    plt.close("all")

    sigma = 150

    av_bkg = np.average(n)

    par, pcov = curve_fit(gauss, b[:-1], n, p0=[max(n), cp_pos, sigma, av_bkg])
    perr = np.sqrt(np.diag(pcov))
    vis_er = par[0] / par[3] ** 2 * 100 * perr[-1]
    # par, covariance = curve_fit(gauss, b[:-1], n, p0=[500, 3000, sigma, 1500])
    fit_plot = gauss(b, par[0], par[1], par[2], par[3])

    st_dev = stdev(data_to_plot)

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
    plt.xlabel(r"$\Delta$ [ps]")
    plt.ylabel("Timestamps [-]")
    # plt.hist(data_to_plot, bins=100, color=chosen_color, label="data")
    # plt.plot(b[:-1], n, "o", color=chosen_color, label="data")
    plt.step(
        b[:-1],
        n,
        color=chosen_color,
        label="data",
    )
    plt.plot(
        b,
        fit_plot,
        "-",
        color="teal",
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
        os.chdir("results/gauss_fit")
    except Exception:
        os.makedirs("results/gauss_fit")
        os.chdir("results/gauss_fit")

    plt.savefig(
        "{file}_pixels"
        "{pix1},{pix2}_fit.png".format(
            file=file_name, pix1=pix_pair[0], pix2=pix_pair[1]
        )
    )

    plt.pause(0.1)
    os.chdir("../..")
