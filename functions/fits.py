""" Module with scripts for calculating and plotting fits of the peaks in
the timestamp differences for LinoSPAD2 data.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * gauss_fit - function for fitting the peak in the timestamp differences
    with a gaussian function.

"""

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from functions import unpack as f_up
from functions.calc_diff import calc_diff as cd


def fit_gauss(path, pix, timestamps: int = 512, show_fig: bool = False):
    """
    Function for fitting the peak in the timestamp differences with a
    gaussian function.

    Parameters
    ----------
    path : str
        Path to the data files.
    pix : array-like
        DESCRIPTION.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

    Returns
    -------
    None.

    """
# TODO: tidy up the code
    def gauss(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):
        print(
            "=========================================\n"
            "Fitting with gauss, Working on {}\n"
            "=========================================".format(filename)
        )
        data = f_up.unpack_numpy(filename)

        data_pix = np.zeros((len(pix), len(data[0])))

        plt.rcParams.update({"font.size": 20})
        # if len(pix) > 2:
        #     fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
        # else:
        #     fig = plt.figure(figsize=(14, 14))
        print("\n> > > Fitting with gauss < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                data_pair = np.vstack((data_pix[q], data_pix[w]))

                output = cd(data_pair, timestamps=timestamps, range_left=17e3, range_right=20e3)

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
                    bins = np.arange(np.min(output), np.max(output), 17.857)
                except Exception:
                    continue
                plt.xlabel("\u0394t [ps]")
                plt.ylabel("Timestamps [-]")
                n, b, p = plt.hist(output, bins=bins, color=chosen_color)

                try:
                    n_max = np.argmax(n)
                    arg_max = (bins[n_max] + bins[n_max + 1]) / 2
                except Exception:
                    arg_max = None
                    pass
                sigma = 200

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
                    label="fit\n" "\u03C3={}".format(par[-1]),
                )
                plt.legend(loc="best")

                try:
                    os.chdir("results/gauss_fit")
                except Exception:
                    os.mkdir("results/gauss_fit")
                    os.chdir("results/gauss_fit")
                plt.savefig(
                    "{file}_pixels"
                    "{pix1}-{pix2}_fit.png".format(
                        file=filename, pix1=pix[q], pix2=pix[w]
                    )
                )
                plt.pause(0.1)
                os.chdir("../..")
