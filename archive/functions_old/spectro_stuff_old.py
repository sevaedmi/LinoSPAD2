"""Module for plotting data from spectrometer setup with LinoSPAD2.

A set of functions to unpack data, count the timestamps, and plot the
results.

This file can also be imported as a module and contains the following
functions:

    * ar_spec - unpacks data, counts the number of timestamps in each
    pixel, fits with gaussian each discovered peak and plots the results.
    Works with firmware version '2212b'.

    * spdc_ac_save - unpacks data, counts the number of timestamps in each
    pixel, collects timestamps differences for an anti-correlation plot
    and plots the results.

    * spdc_ac_cp - plot an anti-correlation plot using SPDC data from
    a .csv file.

"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal as sg
from scipy.optimize import curve_fit
from tqdm import tqdm

from functions import unpack as f_up


def ar_spec(path, board_number: str, tmrl: list, timestamps: int = 512):
    """Plot and fit a spectrum.

    Unpacks spectrum data, plots the number of counts vs wavelength and
    fits with gaussian function each of the peaks. Peaks are looked for
    automatically using a threshold of 10% of max of all counts. Works
    only with LinoSPAD2 firmware version 2212b.

    Parameters
    ----------
    path : str
        Path to datafiles.
    board_number : str
        LinoSPAD2 daughterboard number. Either 'A5' or 'NL11' are
        recognized.
    tmrl: list
        NIST values for the two most right lines.
    timestamps : int, optional
        Number of timestamps per acqusition cycle per TDC. The default
        is 512.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")
    if len(tmrl) != 2:
        raise ValueError(
            "'tmrp' should include exactly two most right lines" "expected in the plot"
        )
    os.chdir(path)

    files = glob.glob("*.dat*")

    valid_per_pixel = np.zeros(256)

    # For transforming pixel number to TDC number and pixel coordinates
    # in that TDC
    pix_coor = np.arange(256).reshape(64, 4)

    # Count the timestamps in each pixel
    for i in tqdm(range(len(files)), desc="Going through files"):
        data = f_up.unpack_2212_numpy(files[i], board_number, timestamps)

        for i in range(0, 256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            valid_per_pixel[i] += len(np.where(data[tdc].T[1][ind] > 0)[0])

    # Mask the hot/warm pixels
    path_to_back = os.getcwd()
    path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    os.chdir(path_to_mask)
    file_mask = glob.glob("*{}*".format(board_number))[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    for i in mask:
        valid_per_pixel[i] = 0

    v_max = np.max(valid_per_pixel)

    peak_pos = sg.find_peaks(valid_per_pixel, threshold=v_max / 10)[0]

    # Convert pixels to wavelengths; NIST values are used, accounting
    # for air refractive index of 1.0003
    pixels = np.arange(0, 256, 1)
    # To convert pix to nm, use linear equation y = ax + b
    # a = (y_2 - y_1) / (p-2 - p_1), where y_1, y_2 are pix position in
    # pixels, p_1, p_2 - in nm.
    # b = y_2 - (y_2 - y_1) / (p_2 - p_1) * p_1 = y_2 - a * p_1
    nm_per_pix = (tmrl[1] / 1.0003 - tmrl[0] / 1.0003) / (peak_pos[-1] - peak_pos[-2])
    x_nm = nm_per_pix * pixels + tmrl[1] / 1.0003 - nm_per_pix * peak_pos[-1]

    peak_pos_nm = (
        np.array(peak_pos) * nm_per_pix + tmrl[1] / 1.0003 - nm_per_pix * peak_pos[-1]
    )

    def gauss(x, A, x0, sigma, C):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

    # Sigma guess in nm
    sigma = 0.1
    # Background level
    valid_bckg = np.copy(valid_per_pixel)
    for i in range(len(peak_pos)):
        valid_bckg[peak_pos[i] - 3 : peak_pos[i] + 3] = 0

    av_bkg = np.average(valid_bckg)

    # Prepare array for each peak
    valid_per_pixel_cut = np.zeros((len(peak_pos), len(valid_per_pixel)))
    fit_plot = np.zeros((len(peak_pos), len(valid_per_pixel)))
    par = np.zeros((len(peak_pos), 4))
    pcov = np.zeros((len(peak_pos), 4, 4))
    perr = np.zeros((len(peak_pos), 4))

    for i in range(len(valid_per_pixel_cut)):
        valid_per_pixel_cut[i] = np.copy(valid_per_pixel)
        for j in np.delete(np.arange(len(peak_pos)), i):
            valid_per_pixel_cut[i][peak_pos[j] - 5 : peak_pos[j] + 5] = av_bkg

    # Fit each peak
    for i in range(len(peak_pos)):
        par[i], pcov[i] = curve_fit(
            gauss,
            x_nm,
            valid_per_pixel_cut[i],
            p0=[max(valid_per_pixel_cut[i]), peak_pos_nm[i], sigma, av_bkg],
        )
        perr[i] = np.sqrt(np.diag(pcov[i]))
        fit_plot[i] = gauss(x_nm, par[i][0], par[i][1], par[i][2], par[i][3])

    colors1 = [
        "#cd9bd8",
        "#da91c5",
        "#e189ae",
        "#e48397",
        "#e08080",
        "#ffda9e",
    ]

    plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 22})
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Counts [-]")
    plt.minorticks_on()
    plt.plot(x_nm, valid_per_pixel, "o-", color="steelblue", label="Data")
    for i in range(len(peak_pos)):
        plt.plot(
            x_nm[peak_pos[i] - 10 : peak_pos[i] + 10],
            fit_plot[i][peak_pos[i] - 10 : peak_pos[i] + 10],
            color=colors1[i],
            linewidth=2,
            label="\n"
            "\u03C3={p1} nm\n"
            "\u03BC={p2} nm".format(
                p1=format(par[i][2], ".3f"), p2=format(par[i][1], ".3f")
            ),
        )
    plt.legend(loc="best", fontsize=16)
    plt.tight_layout()

    try:
        os.chdir("results")
    except Exception:
        os.makedirs("results")
        os.chdir("results")
    plt.savefig(
        "{p1}-{p2} nm.png".format(
            p1=format(peak_pos_nm[0], ".0f"), p2=format(peak_pos_nm[-1], ".0f")
        )
    )
    plt.pause(0.1)
    os.chdir("..")


def spdc_ac_save(
    path,
    board_number: str,
    pix_left: list,
    pix_right: list,
    rewrite: bool,
    timestamps: int = 512,
    delta_window: float = 50e3,
):
    """Calculate and save to csv timestamps differences from SPDC data.

    Unpack SPDC data, calculates timestamp differences between the two
    given lists of pixels' numbers, and saves it to a .csv file. Works
    only with LinoSPAD2 firmware version 2212b.

    Parameters
    ----------
    path : str
        Path to datafiles.
    board_number : str
        LinoSPAD2 daughterboard number. Either 'A5' or 'NL11' are
        recognized.
    pix_left : list
        List of pixel numbers covering signal/idler.
    pix_right : list
        List of pixel numbers covering idler/signal.
    rewrite : bool
        Switch for rewriting the .csv file with timestamp differences.
    timestamps : int, optional
        Number of timestamps per pixel number per TDC. The default is
        512.
    delta_window : float, optional
        Time range in which timestamp differences are collcted. The
        default is 10e3.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean, 'True' for rewriting the .csv")

    os.chdir(path)

    files = glob.glob("*.dat*")

    out_file_name = files[0][:-4] + "-" + files[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{}.csv".format(out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file already exists and will be" "rewritten. ! ! !\n"
                )
                for i in range(5):
                    print("\n! ! ! Deleting the file in {} ! ! !\n".format(5 - i))
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                sys.exit(
                    "\n csv file already exists, 'rewrite' set to" "'False', exiting."
                )
        os.chdir("..")
    except FileNotFoundError:
        pass

    # If .csv file does not exist or it set to be rewritten, prepare
    # header with all pixels' pair combinations
    new_csv = {}

    for q in pix_left:
        for w in pix_right:
            new_csv["{},{}".format(q, w)] = []

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        os.mkdir("delta_ts_data")
        os.chdir("delta_ts_data")

    new_csv_df = pd.DataFrame.from_dict(new_csv)
    new_csv_df.to_csv("{}.csv".format(out_file_name), index=False)

    os.chdir("..")

    # Prepare arrays for the for loop: first for pixel addresses for
    # converting to TDC number and pix number in that TDC (0, 1, 2, 3);
    # second for number of timestamp differences in each pixel pair
    pix_coor = np.arange(256).reshape(64, 4)

    # Mask the hot/warm pixels
    path_to_back = os.getcwd()
    path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    os.chdir(path_to_mask)
    file_mask = glob.glob("*{}*".format(board_number))[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    for i in tqdm(range(len(files)), desc="Going through files"):
        data_all = f_up.unpack_2212_numpy(
            files[i], board_number="A5", timestamps=timestamps
        )

        # For anti-correlation plot
        deltas_all = {}

        # Calculate and collect timestamp differences
        for q in pix_left:
            for w in pix_right:
                if "{},{}".format(q, w) not in list(deltas_all.keys()):
                    deltas_all["{},{}".format(q, w)] = []
                if q in mask or w in mask:
                    continue
                # Find ends of cycles
                cycler = np.argwhere(data_all[0].T[0] == -2)
                # TODO: most probably losing first delta t due to cycling
                cycler = np.insert(cycler, 0, 0)
                # First pixel in the pair
                tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
                pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                # Second pixel in the pair
                tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                # Get timestamp for both pixels in the given cycle
                for cyc in np.arange(len(cycler) - 1):
                    pix1_ = pix1[
                        np.logical_and(pix1 > cycler[cyc], pix1 < cycler[cyc + 1])
                    ]
                    if not np.any(pix1_):
                        continue
                    pix2_ = pix2[
                        np.logical_and(pix2 > cycler[cyc], pix2 < cycler[cyc + 1])
                    ]
                    if not np.any(pix2_):
                        continue
                    # Calculate delta t
                    tmsp1 = data_all[tdc1].T[1][
                        pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                    ]
                    tmsp2 = data_all[tdc2].T[1][
                        pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                    ]
                    for t1 in tmsp1:
                        deltas = tmsp2 - t1
                        ind = np.where(np.abs(deltas) < delta_window)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])

        # Save data as a .csv file
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")
        data_for_plot_df.to_csv(
            "{}.csv".format(out_file_name), mode="a", header=False, index=False
        )
        os.chdir("..")


def spdc_ac_cp(
    path,
    rewrite: bool,
    interpolation: bool = False,
    show_fig: bool = False,
    delta_window: int = 10e3,
):
    """Plot anti-correlation plot from SPDC data.

    Using timestamp differences from .csv file, plot an anti-correlation
    plot.


    Parameters
    ----------
    path : str
        Path to datafiles.
    rewrite : bool
        Switch for overwriting the plot if it exists.
    interpolation : bool, optional
        Switch for applying bessel interpolation on the plot. The
        default is False.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

    Raises
    ------
    TypeError
        Raised if 'rewrite' is not boolean.

    Returns
    -------
    None.

    """
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean, 'True' for rewriting the plot")

    os.chdir(path)
    files_all = glob.glob("*.dat*")
    csv_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # Check if plot exists and if it should be rewrited
    try:
        os.chdir("results/delta_t")
        if os.path.isfile("{name}_delta_t_grid.png".format(name=csv_file_name)):
            if rewrite is True:
                print("\n! ! ! Plot already exists and will be" "rewritten. ! ! !\n")
            else:
                sys.exit("\nPlot already exists, 'rewrite' set to 'False'," "exiting.")
        os.chdir("../..")
    except FileNotFoundError:
        pass

    # Go through plot settings
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    if interpolation is True:
        interpolation = "bessel"
    else:
        interpolation = "None"

    # Matrix of 256 by 256 pixels with number of timestamp differences
    # per pair of pixels.
    mat = np.zeros((256, 256))

    data = pd.read_csv("delta_ts_data/{}.csv".format(csv_file_name))

    # Fill the matrix
    for i in range(len(data.columns)):
        pixs = data.columns[i].split(",")
        ind = np.where(np.abs(np.array(data[data.columns[i]])) < delta_window)[0]
        if not np.any(ind):
            continue
        mat[int(pixs[0])][int(pixs[1])] = len(data[data.columns[i]][ind].dropna())

    # Find where the data in the matrix is for setting limits for plot
    positives = np.where(mat > 0)

    deg = (
        np.arctan(
            (positives[1][-1] - positives[1][0]) / (positives[0][-1] - positives[0][0])
        )
        * 180
        / np.pi
    )

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Pixel [-]")
    plt.xlim(positives[0][0] - 5, positives[0][-1] + 5)
    plt.ylim(positives[1][0] - 5, positives[1][-1] + 5)
    plt.title(
        "SPDC anti-correlation plot\nAngle is {deg} \u00b0".format(
            deg=format(deg, ".2f")
        )
    )
    pos = ax.imshow(mat.T, cmap="cividis", interpolation=interpolation, origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(pos, cax=cax, label="# of coincidences [-]")
    plt.tight_layout()
    try:
        os.chdir("results")
    except Exception:
        os.makedirs("results")
        os.chdir("results")
    plt.savefig("SPDC anti-correlation plot.png")
