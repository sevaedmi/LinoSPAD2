"""Module for working with the calibration data.

This file can also be imported as a module and contains the following
functions:

    * calibrate_TDC_save - calculate a calibration matrix of the TDC
    calibrations and save it as a '.csv' table.

    * unpack_for_offset - unpack binary data applying TDC calibration
    in the process. Used for calculations of the offset calibration.
    
    * delta_save_for_offset - calculate and save timestamps differences
    for pairs of pixels 0, 4 to 255, and 1 to 3 for finding the
    delta t peaks for calculating the offset calibration.
    
    * calib_offset_save - calculate and save the 256 offset compensations
    for all pixels of the given LinoSPAD2 sensor half. The output is 
    saved as a .npy file for later use.

    * calibrate_load - load the calibration matrix from a '.csv' table.
"""
import glob
import os
import sys
from math import ceil
from tqdm import tqdm
from scipy.optimize import curve_fit
import time

import numpy as np
import pandas as pd


def calib_TDC_save(
    path, db_num: str, mb_num: str, fw_ver: str, timestamps: int = 512
):
    """Calculate and save calibration data.

    Function for calculating the calibration matrix and saving it into a
    '.csv' file. The data file used for the calculation should be taken
    with the sensor uniformly illuminated by ambient light.

    Parameters
    ----------
    #TODO
    path : str
        Path to the data file.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The
        default is 512.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(db_num, str) is not True:
        raise TypeError("'db_num' should be string, 'NL11' or 'A5'")
    if isinstance(mb_num, str) is not True:
        raise TypeError("'mb_num' should be string")
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2208' or '2212b'")

    os.chdir(path)
    filename = glob.glob("*.dat*")[0]

    if fw_ver == "2208":
        # read data by 32 bit words
        rawFile = np.fromfile(filename, dtype=np.uint32)
        # lowest 28 bits are the timestamp; convert to ps
        data = (rawFile & 0xFFFFFFF).astype(int) % 140
        # mask nonvalid data with '-1'; 0x80000000 - the 31st, validity bin
        data[np.where(rawFile < 0x80000000)] = -1
        # number of acquisition cycles
        cycles = int(len(data) / timestamps / 256)

        data_matrix = (
            data.reshape(cycles, 256, timestamps)
            .transpose((1, 0, 2))
            .reshape(256, timestamps * cycles)
        )

        # calibration matrix
        cal_mat = np.zeros((256, 140))
        bins = np.arange(0, 141, 1)

        for i in range(256):
            # sort the data into 140 bins
            counts, bin_edges = np.histogram(data_matrix[i], bins=bins)
            # redefine the bin edges using the bin population from above
            cal_mat[i] = np.cumsum(counts) / np.cumsum(counts).max() * 2500

        cal_mat_df = pd.DataFrame(cal_mat)
        cal_mat_df.to_csv(
            "TDC_{db}_{mb}_{fw_ver}.csv".format(
                db=db_num, mb=mb_num, fw_ver=fw_ver
            )
        )

    elif fw_ver == "2212b":
        # read data by 32 bit words
        rawFile = np.fromfile(filename, dtype=np.uint32)
        # lowest 28 bits are the timestamp; convert to ps
        data_t = (rawFile & 0xFFFFFFF).astype(int) % 140
        # pix adress in the given TDC is 2 bits above timestamp
        data_p = ((rawFile >> 28) & 0x3).astype(np.longlong)
        data_t[np.where(rawFile < 0x80000000)] = -1
        # number of acquisition cycle in each datafile
        cycles = int(len(data_t) / timestamps / 65)
        # transofrm into matrix 65 by cycles*timestamps
        data_matrix_p = (
            data_p.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, timestamps * cycles)
        )

        data_matrix_t = (
            data_t.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, timestamps * cycles)
        )

        # cut the 65th TDC that does not hold any actual data from pixels
        data_matrix_p = data_matrix_p[:-1]
        data_matrix_t = data_matrix_t[:-1]

        data_all = np.stack((data_matrix_p, data_matrix_t), axis=2).astype(
            np.longlong
        )

        # calibration matrix
        cal_mat = np.zeros((256, 140))
        bins = np.arange(0, 141, 1)

        pix_coor = np.arange(256).reshape(64, 4)

        for i in range(256):
            # transform pixel number to TDC number and pixel coordinates in
            # that TDC (from 0 to 3)
            tdc, pix = np.argwhere(pix_coor == i)[0]
            # find data from that pixel
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            # cut non-valid timestamps ('-1's)
            ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
            if not np.any(ind):
                continue

            counts, bin_edges = np.histogram(
                data_all[tdc].T[1][ind], bins=bins
            )
            cal_mat[i] = np.cumsum(counts) / np.cumsum(counts).max() * 2500

        cal_mat_df = pd.DataFrame(cal_mat)
        cal_mat_df.to_csv(
            "TDC_{db}_{mb}_{fw_ver}.csv".format(
                db=db_num, mb=mb_num, fw_ver=fw_ver
            )
        )


def unpack_for_offset(
    file,
    db_num: str,
    mb_num: str,
    fw_ver: str,
    timestamps: int = 512,
):
    """Unpack data from firmware version 2212.

    Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.
    Uses Numpy to achieve the best speed for unpacking. Data is returned
    as a 3D array where rows are TDC numbers, columns are the data, and
    each cell contains a pixel number in the TDC (from 0 to 3) and the
    timestamp recorded by that pixel. TDC calibration is applied as it
    is necessary for the offset calibration to succeed.

    Parameters
    ----------
    file : str
        '.dat' data file.
    db_num : str
        LinoSPAD2 daughterboard number.
    mb_num : str
        LinoSPAD2 motherboard (FPGA) number.
    fw_ver : str
        LinoSPAD2 firmware version. Either '2212s' or '2212b' are accepted.
    timestamps : int, optional
        Number of timestamps per TDC per acquisition cycle. The default
        is 512.

    Raises
    ------
    TypeError
        Controller for the type of 'db_num', 'mb_num', and 'fw_ver'
        parameters which should be a string.
    FileNotFoundError
        Controller for stopping the script in the case no calibration
        data were found.

    Returns
    -------
    data_all : array-like
        3D array of pixel coordinates in the TDC and the timestamps.

    """
    # parameter type check
    if isinstance(db_num, str) is not True:
        raise TypeError("'db_num' should be string, 'NL11' or 'A5'")
    if isinstance(mb_num, str) is not True:
        raise TypeError("'db_num' should be string")
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212s' or '2212b'")

    # unpack binary data
    rawFile = np.fromfile(file, dtype=np.uint32)
    # timestamps are lower 28 bits
    data_t = (rawFile & 0xFFFFFFF).astype(np.longlong)
    # pix adress in the given TDC is 2 bits above timestamp
    data_p = ((rawFile >> 28) & 0x3).astype(np.longlong)
    data_t[np.where(rawFile < 0x80000000)] = -1
    # number of acquisition cycle in each datafile
    cycles = int(len(data_t) / timestamps / 65)
    # transofrm into matrix 65 by cycles*timestamps
    data_matrix_p = (
        data_p.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )

    data_matrix_t = (
        data_t.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )
    # cut the 65th TDC that does not hold any actual data from pixels
    data_matrix_p = data_matrix_p[:-1]
    data_matrix_t = data_matrix_t[:-1]
    # insert '-2' at the end of each cycle
    data_matrix_p = np.insert(
        data_matrix_p,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )

    data_matrix_t = np.insert(
        data_matrix_t,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )
    # combine both matrices into a single one, where each cell holds pix
    # coordinates in the TDC and the timestamp
    data_all = np.stack((data_matrix_p, data_matrix_t), axis=2).astype(
        np.longlong
    )
    # if app_calib is False:
    #     data_all[:, :, 1] = data_all[:, :, 1] * 17.857

    # else:
    # path to the current script, two levels up (the script itself is in
    # the path) and two levels down to the calibration data
    pix_coor = np.arange(256).reshape(64, 4)
    path_calib_data = (
        os.path.realpath(__file__) + "/../.." + "/params/calibration_data"
    )

    try:
        cal_mat = calibrate_load(
            path_calib_data, db_num, mb_num, fw_ver, inc_offset=False
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "No .csv file with the calibration data was found, "
            "check the path or run the calibration."
        )

    for i in range(256):
        # transform pixel number to TDC number and pixel coordinates in
        # that TDC (from 0 to 3)
        tdc, pix = np.argwhere(pix_coor == i)[0]
        # find data from that pixel
        ind = np.where(data_all[tdc].T[0] == pix)[0]
        # cut non-valid timestamps ('-1's)
        ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
        if not np.any(ind):
            continue

        data_all[tdc].T[1][ind] = (
            data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140
        ) * 17.857 + cal_mat[i, (data_all[tdc].T[1][ind] % 140)]

    return data_all


def delta_save_for_offset(
    path,
    pixels: list,
    rewrite: bool,
    db_num: str,
    mb_num: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
):
    """Calculate and save timestamp differences into '.csv' file.

    Unpacks data, calculates timestamp differences for the requested
    pixels and saves them into a '.csv' table. Works with firmware
    version 2212. Calculates delta ts with a TDC calibration applied
    for calculations of offset compensations.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.csv' file if it already exists.
    db_num : str
        LinoSPAD2 daughterboard number.
    mb_num : str
        LinoSPAD2 motherboard (FPGA) number.
    fw_ver: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of 'db_num',
        'mb_num', and 'fw_ver' are accepted. The first error is raised
        so that the plot does not accidentally get rewritten in the
        case no clear input was given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(fw_ver, str) is False:
        raise TypeError("'fw_ver' should be string, '2212b' or '2208'")
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(db_num, str) is False:
        raise TypeError("'db_num' should be string, either 'NL11' or 'A5'")

    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("offset_deltas")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
                )
                for i in range(5):
                    print(
                        "\n! ! ! Deleting the file in {} ! ! !\n".format(5 - i)
                    )
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                sys.exit(
                    "\n csv file already exists, 'rewrite' set to"
                    "'False', exiting."
                )
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .csv in a cycle < < <\n"
    )
    if fw_ver == "2212s":
        # for transforming pixel number into TDC number + pixel
        # coordinates in that TDC
        pix_coor = np.arange(256).reshape(4, 64).T
    elif fw_ver == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Mask the hot/warm pixels
    path_to_back = os.getcwd()
    path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    os.chdir(path_to_mask)
    file_mask = glob.glob("*{}*".format(db_num))[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    # Check if 'pixels' is one or two peaks, swap their positions if
    # needed
    if isinstance(pixels[0], list) is True:
        pixels_left = pixels[0]
        pixels_right = pixels[1]
        # Check if pixels from first list are to the left of the right
        # (peaks are not mixed up)
        if pixels_left[-1] > pixels_right[0]:
            plc_hld = pixels_left
            pixels_left = pixels_right
            pixels_right = plc_hld
            del plc_hld
    elif isinstance(pixels[0], int) is True:
        pixels_left = pixels
        pixels_right = pixels

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Prepare a dictionary for output
        deltas_all = {}

        # Unpack data for the requested pixels into dictionary
        data_all = unpack_for_offset(file, db_num, mb_num, fw_ver, timestamps)

        # Calculate and collect timestamp differences
        # for q in pixels:
        for q in pixels_left:
            # for w in pixels:
            for w in pixels_right:
                if w <= q:
                    continue
                # if q in mask or w in mask:
                #     continue
                deltas_all["{},{}".format(q, w)] = []
                # find end of cycles
                cycler = np.argwhere(data_all[0].T[0] == -2)
                cycler = np.insert(cycler, 0, 0)
                # first pixel in the pair
                tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
                pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                # second pixel in the pair
                tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                # get timestamp for both pixels in the given cycle
                for cyc in range(len(cycler) - 1):
                    pix1_ = pix1[
                        np.logical_and(
                            pix1 > cycler[cyc], pix1 < cycler[cyc + 1]
                        )
                    ]
                    if not np.any(pix1_):
                        continue
                    pix2_ = pix2[
                        np.logical_and(
                            pix2 > cycler[cyc], pix2 < cycler[cyc + 1]
                        )
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
        # Save data as a .csv file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("offset_deltas")
        except FileNotFoundError:
            os.mkdir("offset_deltas")
            os.chdir("offset_deltas")
        csv_file = glob.glob("*{}.csv*".format(out_file_name))
        if csv_file != []:
            data_for_plot_df.to_csv(
                "Offset_{}.csv".format(out_file_name),
                mode="a",
                index=False,
                header=False,
            )
        else:
            data_for_plot_df.to_csv(
                "Offset_{}.csv".format(out_file_name), index=False
            )
        os.chdir("..")

    if (
        os.path.isfile(path + "/offset_deltas/{}.csv".format(out_file_name))
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as {file}.csv in "
            "{path} < < <".format(
                file=out_file_name,
                path=path + "\offset_deltas",
            )
        )
    else:
        print("File wasn't generated. Check input parameters.")


def calib_offset_save(
    path, db_num: str, mb_num: str, fw_ver: str, timestamps: int = 512
):
    """Calculate offset calibration and save as .npy.

    Calculate offset calibration for all 256 pixels for the given
    motherboard-daughterboard and firmware version.

    Parameters
    ----------
    path : str
        Path to the data files.
    db_num : str
        LinoSPAD2 daughterboard number.
    mb_num : str
        LinoSPAD2 motherboard (FPGA) number.
    fw_ver : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per cycle per TDC, by default 512.
    """

    def gauss(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    # try:
    #     os.chdir(path + r"/delta_ts_data/")
    #     file_csv = glob.glob("*.csv*")[0]
    #     dt_all = np.array(pd.read_csv(file_csv))
    #     print("\n>>> 'csv' file found, working on <<<\n")
    # except (FileNotFoundError, IndexError):
    #     print("\n>>> Collecting delta ts <<<\n")

    # Calculate delta ts for pixels 0 and 4-255
    delta_save_for_offset(
        path,
        pixels=[[0], [x for x in range(1, 256)]],
        db_num=db_num,
        mb_num=mb_num,
        fw_ver=fw_ver,
        rewrite=True,
        timestamps=timestamps,
    )
    os.chdir(path + r"/offset_deltas/")
    file_csv = glob.glob("*Offset_*.csv*")[0]
    dt_all = np.array(pd.read_csv(file_csv))
    os.chdir("..")

    peak_positions_3_256 = np.zeros(256)
    peak_positions_1_4 = np.zeros(256)

    # Fit to find where the peak ends up.
    for i in range(255):
        dt_nonan_arr = dt_all[:, i][~np.isnan(dt_all[:, i])]
        if dt_nonan_arr.size == 0:
            continue
        else:
            bins = np.arange(
                np.min(dt_nonan_arr), np.max(dt_nonan_arr), 17.857
            )

            counts, binEdges = np.histogram(dt_nonan_arr, bins=bins)
            binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])

            n_max = np.argmax(counts)
            arg_max = (binEdges[n_max] + binEdges[n_max + 1]) / 2
            sigma = 200

            parameters, covariance = curve_fit(
                gauss, binCenters, counts, p0=[max(counts), arg_max, sigma]
            )

            peak_positions_3_256[i] = parameters[1]
            peak_positions_3_256[0:3] = 0

    # Calculate delta ts for pixels 1,2,3
    delta_save_for_offset(
        path,
        pixels=[[1, 2, 3], [4]],
        db_num=db_num,
        mb_num=mb_num,
        fw_ver=fw_ver,
        rewrite=True,
        timestamps=timestamps,
    )
    os.chdir(path + r"/offset_deltas/")
    file_csv = glob.glob("*.csv*")[0]
    dt_all = np.array(pd.read_csv(file_csv))
    os.chdir("..")

    peak_positions_1_4 = np.zeros(256)

    # Fit to find where the peak ends up.
    for i in range(3):
        dt_nonan_arr = dt_all[:, i][~np.isnan(dt_all[:, i])]
        if dt_nonan_arr.size == 0:
            continue
        else:
            bins = np.arange(
                np.min(dt_nonan_arr), np.max(dt_nonan_arr), 17.857
            )

            counts, binEdges = np.histogram(dt_nonan_arr, bins=bins)
            binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])

            n_max = np.argmax(counts)
            arg_max = (binEdges[n_max] + binEdges[n_max + 1]) / 2
            sigma = 200

            parameters, covariance = curve_fit(
                gauss, binCenters, counts, p0=[max(counts), arg_max, sigma]
            )

            peak_positions_1_4[i] = parameters[1]

    peak_positions = peak_positions_1_4 + peak_positions_3_256
    print(peak_positions)
    print(peak_positions)

    # Indices for a system of linear equations for offset calculation.
    # Last equation is for setting the average offset equal to zero.
    a = np.zeros((256, 256))
    for i in range(3, 255):
        a[i][0] = 1
        a[i][i + 1] = -1
    a[0][1] = 1
    a[1][2] = 1
    a[2][3] = 1
    a[0][4] = -1
    a[1][4] = -1
    a[2][4] = -1
    a[-1] = 1

    # Solving the system of equations, the result are offsets
    offsets = np.linalg.solve(a, peak_positions)

    np.save("Offset_{}_{}_{}.npy".format(db_num, mb_num, fw_ver), offsets)


def calibrate_load(
    path, db_num: str, mb_num: str, fw_ver: str, inc_offset: bool = True
):
    """Load the calibration data.

    Parameters
    ----------
    path : str
        Path to the '.csv' file with the calibration matrix.
    db_num: str
        The LinoSPAD2 daughterboard number.
    mb_num : str
        LinoSPAD2 motherboard (FPGA) number.
    fw_ver: str
        LinoSPAD2 firmware version.
    inc_offset : bool, optional
        Switch for including the offset calibration. The default is
        True.

    Returns
    -------
    data_matrix : ndarray
        Matrix of 256x140 with the calibrated data.
    offset_arr : array
        Array of 256 offset compensations for all pixels.

    """
    path_to_back = os.getcwd()
    os.chdir(path)

    # Compensating for TDC nonlinearities
    file_TDC = glob.glob(
        "*TDC_{db}_{mb}_{fw}*".format(db=db_num, mb=mb_num, fw=fw_ver)
    )[0]
    # Compensating for offset
    if inc_offset is True:
        try:
            file_offset = glob.glob(
                "*Offset_{db}_{mb}_{fw}*".format(
                    db=db_num, mb=mb_num, fw=fw_ver
                )
            )[0]
        except IndexError:
            raise IndexError(
                "No .npy file with offset calibration data was found"
            )
        offset_arr = np.load(file_offset)

    # Skipping first row of TDC bins' numbers
    data_matrix_TDC = np.genfromtxt(file_TDC, delimiter=",", skip_header=1)
    # Cut the first column which is pixel numbers
    data_matrix_TDC = np.delete(data_matrix_TDC, 0, axis=1)

    os.chdir(path_to_back)

    return (
        (data_matrix_TDC, offset_arr)
        if inc_offset is True
        else data_matrix_TDC
    )
