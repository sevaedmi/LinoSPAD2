"""Older functions for cross-talk calculation and plotting.

"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from scipy.optimize import curve_fit
from scipy.stats import sem
from tqdm import tqdm

from LinoSPAD2.functions import calc_diff as cd
from LinoSPAD2.functions import unpack as f_up
from LinoSPAD2.functions import utils


def collect_cross_talk(
    path,
    pixels,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 10e3,
    step: int = 1,
    include_offset: bool = True,
    apply_calibration: bool = True,
    correct_pixel_addressing: bool = False,
):
    """Calculate cross-talk and save it to a '.csv' file.

    Calculate timestamp differences for all pixels in the given range,
    where all timestamp differences are calculated for the first pixel
    in the range. Works with firmware version "2212s only". The
    output is saved as a '.csv' file in the folder "/cross_talk_data",
    which is created if it does not exist, in the same folder where
    datafiles are located.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pixels : list
        List of pixel numbers. The list should be constructed in such
        way, the the first number is the aggressor pixel.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per pixel per cycle. The default is 512.
    delta_window : float, optional
        A width of a window in which the number of timestamp differences
        are counted. The default value is 10e3 (10ns).
    step : int, optional
        Step for the histogram bins. The default is 1.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    correct_pixel_addressing : bool, optional
        Switch for correcting pixel address. Should be used for FPGA
        board on side 23. The default is False.

    Returns
    -------
    None.

    Notes
    -----
    Cross-talk data should be collected with the '2212s' firmware version
    as in '2212b' numbers don't fully represent reality. This is
    mainly due to how pixels are connected to separate TDCs in the
    '2212b' firmware version.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string")

    # Prepare lists for collecting everything required for cross-talk
    # number calculation
    print("\n> > > Collecting data for cross-talk analysis < < <\n")
    file_name_list = []
    pix1_list = []
    pix2_list = []
    timestamps_list1 = []
    timestamps_list2 = []
    deltas_list = []
    ct_list = []

    os.chdir(path)

    files = glob.glob("*.dat*")

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        print(
            "\nFor firmware version '2212b' cross-talk numbers "
            "would be incorrect, try data collected with '2212s'"
        )
        sys.exit()
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Break the given list of pixels into two: first one is the aggresor,
    # second list - victims only
    pixels = [pixels[0], pixels[1:]]

    for file in tqdm(files):
        data = f_up.unpack_binary_data(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset,
            apply_calibration,
        )

        timestamps_per_pixel = np.zeros(256)

        for i in range(256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            timestamps_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])

        if correct_pixel_addressing:
            fix = np.zeros(len(timestamps_per_pixel))
            fix[:128] = timestamps_per_pixel[128:]
            fix[128:] = np.flip(timestamps_per_pixel[:128])
            timestamps_per_pixel = fix
            del fix

        # Population of the aggressor pixel
        timestamps_pix1 = timestamps_per_pixel[pixels[0]]

        for j in pixels[1]:
            if timestamps_pix1 == 0:
                continue

            # Population of the victim pixel
            timestamps_pix2 = timestamps_per_pixel[j]

            deltas = cd.calculate_differences_2212(
                data, [pixels[0], j], pix_coor
            )

            counts, bin_edges = np.histogram(
                deltas[f"{pixels[0]},{j}"],
                bins=200,
            )
            bin_centers = (bin_edges - 17.857 * step / 2)[1:]

            params, cov = curve_fit(
                utils.gaussian,
                bin_centers,
                counts,
                p0=[
                    max(counts),
                    bin_centers[np.argmax(counts)],
                    100,
                    np.median(counts),
                ],
                # sigma=sigma_values,
            )

            # lower_limit = params[1] - 2 * params[2]
            # upper_limit = params[1] + 2 * params[2]

            plt.figure(figsize=(10, 8))
            plt.plot(bin_centers, counts, "o")
            plt.plot(bin_centers, utils.gaussian(bin_centers, *params))

            # data_arr = np.array(deltas[f"{pixels[0]},{j}"])
            # data_in_interval = [
            #     (data_arr >= lower_limit) & (data_arr <= upper_limit)
            # ]
            # bckg_center_position = params[1] - 7 * params[2]
            # bckg_in_interval = data_arr[
            #     (data_arr > bckg_center_position - 2 * params[2])
            #     & (data_arr < bckg_center_position + 2 * params[2])
            # ]
            # The cross-talk number in %
            ct = (
                len(deltas[f"{pixels[0]},{j}"])
                * 100
                / (timestamps_pix1 + timestamps_pix2)
            )

            file_name_list.append(file)
            pix1_list.append(pixels[0])
            pix2_list.append(j)
            timestamps_list1.append(timestamps_pix1)
            timestamps_list2.append(timestamps_pix2)
            deltas_list.append(len(deltas[f"{pixels[0]},{j}"]))
            ct_list.append(ct)

        print(
            "\n> > > Saving data as 'CT_data_{}-{}.csv' in"
            " {path} < < <\n".format(
                files[0], files[-1], path=path + "/cross_talk_data"
            )
        )

        dic = {
            "File": file_name_list,
            "Pixel 1": pix1_list,
            "Pixel 2": pix2_list,
            "Timestamps 1": timestamps_list1,
            "Timestamps 2": timestamps_list2,
            "Deltas": deltas_list,
            "CT": ct_list,
        }

        cross_talk_data = pd.DataFrame(dic)

        try:
            os.chdir("cross_talk_data")
        except FileNotFoundError:
            os.makedirs("{}".format("cross_talk_data"))
            os.chdir("cross_talk_data")

    # Check if the '.csv' file with cross-talk numbers for these
    # data files already exists
    if (
        glob.glob(
            "*CT_data_{}-{}_pixel_{}.csv*".format(
                files[0], files[-1], pixels[0]
            )
        )
        == []
    ):
        cross_talk_data.to_csv(
            "CT_data_{}-{}_pixel_{}.csv".format(
                files[0], files[-1], pixels[0]
            ),
            index=False,
        )
    else:
        cross_talk_data.to_csv(
            "CT_data_{}-{}_pixel_{}.csv".format(
                files[0], files[-1], pixels[0]
            ),
            mode="a",
            index=False,
            header=False,
        )


def plot_cross_talk(path, pix1, scale: str = "linear"):
    """Plot cross-talk data from a '.csv' file.

    Plots cross-talk data from a '.csv' file as cross-talk values (in %)
    vs distance in pixels from the given pixel to the right. The plot is
    saved in the folder "/results/cross_talk", which is created if it
    does not exist, in the same folder where data are located.

    Parameters
    ----------
    path : str
        Path to the folder where a '.csv' file with the cross-talk data is
        located.
    pix1 : int
        Pixel number relative to which the cross-talk data should be
        plotted - the aggressor pixel.
    scale : str, optional
        Switch for plot scale: logarithmic or linear. Default is "linear".

    Returns
    -------
    None.

    """
    print("\n> > > Plotting cross-talk vs distance in pixels < < <\n")
    os.chdir(path)

    files = glob.glob("*.dat*")
    files.sort(key=os.path.getmtime)

    # os.chdir(path + "/cross_talk_data")

    os.chdir("cross_talk_data")

    file = glob.glob(
        "*CT_data_{}-{}_pixel_{}.csv*".format(files[0], files[-1], pix1)
    )[0]

    plot_name = "{}_{}".format(files[0], files[-1])

    data = pd.read_csv(file)

    distance = []
    ct = []
    yerr = []

    data_cut = data.loc[data["Pixel 1"] == pix1]

    pix2 = data["Pixel 2"].unique()
    pix2 = np.delete(pix2, np.where(pix2 <= pix1)[0])
    pix2 = np.sort(pix2)

    for i, pix in enumerate(pix2):
        ct_pix = data_cut[data_cut["Pixel 2"] == pix].CT.values

        if ct_pix.size <= 0:
            continue

        distance.append(pix - pix1)
        if len(ct_pix) > 1:
            ct.append(np.average(ct_pix))
            yerr.append(sem(ct_pix))
        else:
            ct.append(ct_pix)

        xticks = np.linspace(
            distance[0], distance[-1], int(len(distance) / 10), dtype=int
        )

    plt.rcParams.update({"font.size": 20})

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    if scale == "log":
        plt.yscale("log")
    if not yerr:
        ax1.plot(distance, ct, color="salmon")
    else:
        ax1.errorbar(distance, ct, yerr=yerr, color="salmon")
    ax1.set_xlabel("Distance in pixels [-]")
    ax1.set_ylabel("Average cross-talk [%]")
    ax1.set_title("Pixel {}".format(pix1))
    ax1.set_xticks(xticks)

    try:
        os.chdir("../results/cross_talk")
    except FileNotFoundError:
        os.makedirs("../results/cross_talk")
        os.chdir("../results/cross_talk")

    plt.savefig("{plot}_{pix}.png".format(plot=plot_name, pix=pix1))
