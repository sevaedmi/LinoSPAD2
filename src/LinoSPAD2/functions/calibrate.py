"""Module for working with the calibration data.

This file can also be imported as a module and contains the following
functions:
    #TODO
    * calibrate_save - calculate a calibration matrix and save it as
    a '.csv' table.

    * calibrate_load - load the calibration matrix from a '.csv' table.
"""
import glob
import os

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


def calib_offset_save(
    path, db_num: str, mb_num: str, fw_ver: str, timestamps: int = 512
):
    spdc_ac_save(
        path=path,
        board_number=board_number,
        pix_left=np.arange(0, 1),
        pix_right=np.arange(1, 256),
        rewrite=True,
        timestamps=timestamps,
        delta_window=30e3,
    )

    def gauss(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    peak_positions_4_256 = np.zeros(256)

    for i in range(255):
        dt_nonan_arr = dt_arr_4_256[:, i][~np.isnan(dt_arr_4_256[:, i])]
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

            fit_y = gauss(
                binCenters, parameters[0], parameters[1], parameters[2]
            )
            peak_positions[i] = parameters[1]
            peak_positions[0:3] = 0

    peak_positions_1_3 = np.zeros(256)
    for i in range(3):
        dt_nonan_arr = dt_arr_1_3[:, i][~np.isnan(dt_arr_1_3[:, i])]
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

            fit_y = gauss(
                binCenters, parameters[0], parameters[1], parameters[2]
            )
            peak_positions_others[i] = parameters[1]

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

    offsets = np.linalg.solve(a, peak_positions_fin)

    np.save("A5_offsets_arr.npy", offsets)


def calibrate_load(
    path, db_num: str, mb_num: str, fw_ver: str, inc_offset: bool = True
):
    """Load the calibration data.

    Parameters
    ----------
    path : str
        Path to the '.csv' file with the calibration matrix.
    db_num: str
        The LinoSPAD2 board number.
    fw_ver: str
        LinoSPAD2 firmware version.

    Returns
    -------
    data_matrix : ndarray
        Matrix of 256x140 with the calibrated data.

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
