"""Module with function for plotting data from spectrometer setup with LinoSPAD2.

A set of functions to unpack data, count the timestamps, and plot the results.

This file can also be imported as a module and contains the following
functions:

    * ar_spec - unpacks data, counts the number of timestamps in each pixel,
    fits with gaussian each discovered peak and plots the results.

    * spdc_ac - unpacks data, counts the number of timestamps in each pixel,
    collects timestamps differences for an anti-correlation plot and plots
    the results.

"""
import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg
from scipy.optimize import curve_fit
from tqdm import tqdm

from functions import unpack as f_up


def ar_spec(path, board_number: str, tnl: list, timestamps: int = 512):
    """Plot and fit a spectrum.

    Unpacks spectrum data, plots the number of counts vs wavelength and fits
    with gaussian function each of the peaks. Peaks are looked for automatically using
    a threshold of 10% of max of all counts. Works only with LinoSPAD2 firmware version
    2212b.

    Parameters
    ----------
    path : str
        Path to datafiles.
    board_number : str
        LinoSPAD2 daughterboard number. Either 'A5' or 'NL11' are recognized.
    tnl: list
        NIST values for two neighboring lines.
    timestamps : int, optional
        Number of timestamps per acqusition cycle per TDC. The default is 512.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")
    if len(tnl) != 2:
        raise ValueError(
            "'tmrp' should include exactly two most right lines expected in" "the plot"
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
    path_to_mask = os.path.realpath(__file__) + "/../.." + "/masks"
    os.chdir(path_to_mask)
    file_mask = glob.glob("*{}*".format(board_number))[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    for i in mask:
        valid_per_pixel[i] = 0

    v_max = np.max(valid_per_pixel)

    peak_pos = sg.find_peaks(valid_per_pixel, threshold=v_max / 10)[0]

    # Convert pixels to wavelengths; NIST values are used, accounting for air refractive
    # index of 1.0003
    pixels = np.arange(0, 256, 1)
    nm_per_pix = (tnl[1] / 1.0003 - tnl[0] / 1.0003) / (peak_pos[-1] - peak_pos[-2])
    x_nm = nm_per_pix * pixels + tnl[1] / 1.0003 - nm_per_pix * peak_pos[-1]

    peak_pos_nm = (
        np.array(peak_pos) * nm_per_pix + tnl[1] / 1.0003 - nm_per_pix * peak_pos[-1]
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

    # colors = ["#008080", "#009480", "#00a880", "#00bc80", "#00d080", "#00e480"]
    colors1 = ["#cd9bd8", "#da91c5", "#e189ae", "#e48397", "#e08080", "#ffda9e"]

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
    plt.savefig("{p1}-{p2} nm.png".format(p1=peak_pos[0], p2=peak_pos[-1]))
    plt.pause(0.1)
    os.chdir("..")


# TODO: convert pix numbers to nm
def spdc_ac(
    path,
    board_number: str,
    pix_left: list,
    pix_right: list,
    interpolation: bool = False,
    timestamps: int = 512,
    delta_window: float = 10e3,
):
    """Plot counts and anti-correlation plot for SPDC data.

    Unpack SPDC data and plot counts vs pixel number and an anti-correlation plot.
    Works only with LinoSPAD2 firmware version 2212b.

    Parameters
    ----------
    path : str
        Path to datafiles.
    board_number : str
        LinoSPAD2 daughterboard number. Either 'A5' or 'NL11' are recognized.
    pix_left : list
        List of pixel numbers covering signal/idler.
    pix_right : list
        List of pixel numbers covering idler/signal.
    interpolation: bool, optional
        Switch for interpolating the anti-correlation plot. Default is False.
    timestamps : int, optional
        Number of timestamps per pixel number per TDC. The default is 512.
    delta_window : float, optional
        Time range in which timestamp differences are collcted. The default is 10e3.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, either 'NL11' or 'A5'")
    if isinstance(interpolation, bool) is not True:
        raise TypeError("'interpolation' should be boolean.")

    os.chdir(path)

    files = glob.glob("*.dat*")

    valid_per_pixel = np.zeros(256)

    pix_coor = np.arange(256).reshape(64, 4)

    mat = np.zeros((256, 256))

    for i, file in enumerate(files):
        data_all = f_up.unpack_2212_numpy(file, board_number="A5", timestamps=20)

        # For plotting counts
        for i in np.arange(0, 256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            ind1 = ind[np.where(data_all[tdc].T[1][ind] > 0)[0]]
            valid_per_pixel[i] += len(data_all[tdc].T[1][ind1])

        # For anti-correlation plot
        deltas_all = {}

        # Calculate and collect timestamp differences
        for q in pix_left:
            for w in pix_right:
                deltas_all["{},{}".format(q, w)] = []

                # find end of cycles
                cycler = np.argwhere(data_all[0].T[0] == -2)
                # TODO: most probably losing first delta t due to cycling
                cycler = np.insert(cycler, 0, 0)
                # first pixel in the pair
                tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
                pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                # second pixel in the pair
                tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                # get timestamp for both pixels in the given cycle
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
                    # calculate delta t
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
                        mat[q][w] += len(ind)

    # background data for subtracting
    path_bckg = path + "/bckg"
    os.chdir(path_bckg)

    files = glob.glob("*.dat*")

    valid_per_pixel_bckg = np.zeros(256)

    for i, file in enumerate(files):
        data_all_bckg = f_up.unpack_2212_numpy(file, board_number="A5", timestamps=20)

        # Fot plotting counts
        for i in np.arange(0, 256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data_all_bckg[tdc].T[0] == pix)[0]
            ind1 = ind[np.where(data_all_bckg[tdc].T[1][ind] > 0)[0]]
            valid_per_pixel_bckg[i] += len(data_all_bckg[tdc].T[1][ind1])

    os.chdir("..")

    # Mask the hot/warm pixels
    path_to_back = os.getcwd()
    path_to_mask = os.path.realpath(__file__) + "/../.." + "/masks"
    os.chdir(path_to_mask)
    file_mask = glob.glob("*{}*".format(board_number))[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    for i in mask:
        valid_per_pixel[i] = 0
        valid_per_pixel_bckg[i] = 0

    plt.ion()
    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(10, 7))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Counts [-]")
    plt.title("SPDC data, background subtracted")
    plt.plot(valid_per_pixel - valid_per_pixel_bckg, "o-", color="teal")
    plt.show()

    try:
        os.chdir("results")
    except Exception:
        os.makedirs("results")
        os.chdir("results")
    plt.savefig("SPDC counts.png")
    plt.pause(0.1)
    os.chdir("..")

    if interpolation is True:
        interpolation = "bessel"
    else:
        interpolation = "None"

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Pixel [-]")
    plt.title("SPDC anti-correlation plot")
    pos = ax.imshow(mat.T, cmap="cividis", interpolation=interpolation, origin="lower")
    fig.colorbar(pos, ax=ax)

    try:
        os.chdir("results")
    except Exception:
        os.makedirs("results")
        os.chdir("results")
    plt.savefig("SPDC anti-correlation plot.png")
    plt.pause(0.1)
    os.chdir("..")
