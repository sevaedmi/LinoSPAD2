"""Module with scripts for unpacking data from LinoSPAD2.

Functions for unpacking either 'txt' of 'dat' data files of LinoSPAD2.
Functions for either 10, 512 or a given number of timestamps per acquisition
cycle per pixel are available.

This file can also be imported as a module and contains the following
functions:

    * unpack_binary_flex - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle

    * unpack_numpy - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle. Uses the calibration data. Imputing the
    LinoSPAD2 board number is required.

    * unpack_numpy_dict - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle. Uses the calibration data. Imputing the
    LinoSPAD2 board number is required. Returns a dictionary of timestamps
    from the requested pixels.

    * unpack_2212 - function for unpacking data into a dictionary for the
    LinoSPAD2 firmware 2212 versions "skip" and "block".

    * unpack_2212_numpy - function for unpacking data from LinoSPAD2, firmware
    version '2212b'. Utilizes the numpy library to speed up the process.


"""

import os
import sys
from glob import glob
from struct import unpack

import numpy as np

from functions.calibrate import calibrate_load


def unpack_binary_flex(filename, timestamps: int = 512):
    """Unpack binary data from the LinoSPAD2.

    Unpacking a single 'dat' output of the LinoSPAD2. Due to the straightforward
    approach used mainly for debugging, otherwise is pretty slow.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely timestamps lines
        of data per acquistion cycle is written.
    timestamps: int, optional
        Number of binary-encoded timestamps in the 'dat' file. The default
        value is 512.

    Returns
    -------
    data_matrix : array_like
        A 2D matrix (256 pixels by timestamps X number-of-cycles) of
        timestamps.

    """
    timestamp_list = []
    address_list = []

    with open(filename, "rb") as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:  # check validity bit: if 1
                # - timestamp is valid
                timestamp = packet[0] & 0xFFFFFFF  # cut the higher bits,
                # leave only timestamp ones
                # 2.5 ns from TDC 400 MHz clock read out 140 bins from 35
                # elements of the delay line - average bin size is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
            # not in this firmware??
            address_list.append(address)
    # rows=#pixels, cols=#cycles
    data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))

    noc = len(timestamp_list) / timestamps / 256  # number of cycles,
    # timestamps data lines per pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][
                k * timestamps : k * timestamps + timestamps
            ] = timestamp_list[
                (i + 256 * k) * timestamps : (i + 256 * k) * timestamps + timestamps
            ]
            i = i + 1
        k = k + 1
    return data_matrix


def unpack_numpy(
    filename,
    board_number: str,
    timestamps: int = 512,
    pix: list = [],
    app_mask: bool = True,
):
    """Unpack binary data from LinoSPAD2.

    Function for unpacking the .dat data files using the calibration
    data. The output is a matrix of '256 x timestamps*number_of_cycles'
    timestamps in ps. The fastest version that utilizes the numpy library.
    Works only with firmware version 2208.

    Parameters
    ----------
    filename : str
        Name of the .dat file.
    board_number : str
        LinoSPAD2 board number.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.
    pix : list, optional
        List of pixel numbers for which the data should be returned. The default is [],
        in which case data for all pixels will be returned.

    Returns
    -------
    data_matrix : ndarray
        Matrix of '256 x timestamps*number_of_cycles' timestamps.

    Examples
    --------
    To see how the function works, consider a setup of 4 pixels,
    5 timestamps per pixel per acquisition cycle, 3 cycles total.
    The output of the LinoSPAD2 data acquisition software is pixel
    by pixel, cycle after cycle. Therefore, for timestamps from 0
    to 59, timestamps 0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 40, 41, 42,
    43, 44 are from the first pixel.

    >>> a = np.arange(4*5*3)
    >>> a
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    >>> b = a.reshape(3, 4, 5)
    >>> b
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
           [[20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39]],
           [[40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59]]])
    >>> c = b.transpose((1, 0, 2))
    >>> c
    array([[[ 0,  1,  2,  3,  4],
            [20, 21, 22, 23, 24],
            [40, 41, 42, 43, 44]],
           [[ 5,  6,  7,  8,  9],
            [25, 26, 27, 28, 29],
            [45, 46, 47, 48, 49]],
           [[10, 11, 12, 13, 14],
            [30, 31, 32, 33, 34],
            [50, 51, 52, 53, 54]],
           [[15, 16, 17, 18, 19],
            [35, 36, 37, 38, 39],
            [55, 56, 57, 58, 59]]])
    >>> d = c.reshape(4, 3*5)
    >>> d
    array([[ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44],
           [ 5,  6,  7,  8,  9, 25, 26, 27, 28, 29, 45, 46, 47, 48, 49],
           [10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50, 51, 52, 53, 54],
           [15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59]])

    """
    # read data by 32 bit words
    rawFile = np.fromfile(filename, dtype=np.uint32)
    # lowest 28 bits are the timestamp; convert to longlong, int is not enough
    data = (rawFile & 0xFFFFFFF).astype(np.longlong)
    # mask nonvalid data with '-1'
    data[np.where(rawFile < 0x80000000)] = -1
    # number of acquisition cycles
    cycles = int(len(data) / timestamps / 256)

    data_matrix = (
        data.reshape(cycles, 256, timestamps)
        .transpose((1, 0, 2))
        .reshape(256, timestamps * cycles)
    )
    # path to the current script, two levels up (the script itself is in the path) and
    # one level down to the calibration data
    path_calib_data = os.path.realpath(__file__) + "/../.." + "/calibration_data"

    try:
        cal_mat = calibrate_load(path_calib_data, board_number)
    except FileNotFoundError:
        print(
            "No .csv file with the calibration data was found, check the path "
            "or run the calibration."
        )
        sys.exit()
    for i in range(256):
        ind = np.where(data_matrix[i] >= 0)[0]
        data_matrix[i, ind] = (
            data_matrix[i, ind] - data_matrix[i, ind] % 140
        ) * 17.857 + cal_mat[i, (data_matrix[i, ind] % 140)]

    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/masks"
        os.chdir(path_to_mask)
        file_mask = glob("*{}*".format(board_number))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        data_matrix[mask] = -1
        os.chdir(path_to_back)

    if not np.any(pix):
        pix = np.arange(256)

    data_matrix = data_matrix[pix, :]

    return data_matrix


def unpack_numpy_dict(
    filename,
    board_number: str,
    timestamps: int = 512,
    pix: list = [],
    app_mask: bool = True,
):
    """Unpack binary data from LinoSPAD2.

    Function for unpacking the .dat data files using the calibration
    data. The output is a matrix of '256 x timestamps*number_of_cycles'
    timestamps in ps. The fastest version that utilizes the numpy library.
    Unpacks data only from firmware version 2208.

    Parameters
    ----------
    filename : str
        Name of the .dat file.
    board_number : str
        LinoSPAD2 board number.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    data_matrix : ndarray
        Matrix of '256 x timestamps*number_of_cycles' timestamps.

    Examples
    --------
    To see how the function works, consider a setup of 4 pixels,
    5 timestamps per pixel per acquisition cycle, 3 cycles total.
    The output of the LinoSPAD2 data acquisition software is pixel
    by pixel, cycle after cycle. Therefore, for timestamps from 0
    to 59, timestamps 0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 40, 41, 42,
    43, 44 are from the first pixel.

    >>> a = np.arange(4*5*3)
    >>> a
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    >>> b = a.reshape(3, 4, 5)
    >>> b
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
           [[20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39]],
           [[40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59]]])
    >>> c = b.transpose((1, 0, 2))
    >>> c
    array([[[ 0,  1,  2,  3,  4],
            [20, 21, 22, 23, 24],
            [40, 41, 42, 43, 44]],
           [[ 5,  6,  7,  8,  9],
            [25, 26, 27, 28, 29],
            [45, 46, 47, 48, 49]],
           [[10, 11, 12, 13, 14],
            [30, 31, 32, 33, 34],
            [50, 51, 52, 53, 54]],
           [[15, 16, 17, 18, 19],
            [35, 36, 37, 38, 39],
            [55, 56, 57, 58, 59]]])
    >>> d = c.reshape(4, 3*5)
    >>> d
    array([[ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44],
           [ 5,  6,  7,  8,  9, 25, 26, 27, 28, 29, 45, 46, 47, 48, 49],
           [10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50, 51, 52, 53, 54],
           [15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59]])

    """
    # read data by 32 bit words
    rawFile = np.fromfile(filename, dtype=np.uint32)
    # lowest 28 bits are the timestamp; convert to longlong, int is not enough
    data = (rawFile & 0xFFFFFFF).astype(np.longlong)
    # mask nonvalid data with '-1'
    data[np.where(rawFile < 0x80000000)] = -1
    # number of acquisition cycles
    cycles = int(len(data) / timestamps / 256)

    data_matrix = (
        data.reshape(cycles, 256, timestamps)
        .transpose((1, 0, 2))
        .reshape(256, timestamps * cycles)
    )
    # path to the current script, two levels up (the script itself is in the path) and
    # one level down to the calibration data
    path_calib_data = os.path.realpath(__file__) + "/../.." + "/calibration_data"

    try:
        cal_mat = calibrate_load(path_calib_data, board_number)
    except FileNotFoundError:
        print(
            "No .csv file with the calibration data was found, check the path "
            "or run the calibration."
        )
        sys.exit()
    for i in range(256):
        ind = np.where(data_matrix[i] >= 0)[0]
        data_matrix[i, ind] = (
            data_matrix[i, ind] - data_matrix[i, ind] % 140
        ) * 17.857 + cal_mat[i, (data_matrix[i, ind] % 140)]

    if app_mask is True:
        path_to_back = os.getcwd()
        path_to_mask = os.path.realpath(__file__) + "/../.." + "/masks"
        os.chdir(path_to_mask)
        file_mask = glob("*{}*".format(board_number))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        os.chdir(path_to_back)

    output = {}
    if not np.any(pix):
        pix = np.arange(256)
    for px in pix:
        if px in mask:
            output["{}".format(px)] = np.full(len(data_matrix[px]), -1)
        else:
            output["{}".format(px)] = data_matrix[px]

    ins = np.arange(timestamps, timestamps * (cycles + 1), timestamps)

    for key in output.keys():
        output[key] = np.insert(output[key], ins, -2)
        output[key] = np.delete(output[key], np.where(output[key] == -1))

    return output


# def unpack_mult_cut(files, pixels, board_number: str, timestamps: int = 512):
#     """Unpack binary data from LinoSPAD2 only for given pixels.

#     Returns timestamps only for the given pixels. Uses the calibration data.

#     Parameters
#     ----------
#     files : list
#         List of files' names with the binary data from LinoSPAD2.
#     pixels : array-like or list
#         Array or list of pixel numbers for which the data should be unpacked.
#     board_number : str
#         The LinoSPAD2 daughterboard number.
#     timestamps : int, optional
#         Number of timestamps per acquisition cycle per pixel. The default is 512.

#     Returns
#     -------
#     ndarray-like
#         A matrix of pixels X timestamps*number_of_cycles of timestamps.

#     """
#     pixels = np.sort(pixels)

#     data_all = []

#     for i, file in enumerate(files):
#         if not np.any(data_all):
#             data_all = unpack_numpy(file, board_number, timestamps)
#         else:
#             data_all = np.append(
#                 data_all, unpack_numpy(file, board_number, timestamps), axis=1
#             )

#     output = []

#     for i in range(len(pixels)):
#         if not np.any(output):
#             output = data_all[pixels[0]]
#         else:
#             output = np.vstack((output, data_all[pixels[i]]))

#     return output


def unpack_2212(filename, board_number: str, fw_ver: str, timestamps: int = 512):
    """Unpack binary data from LinoSPAD2, firmware version 2212.

    Function for unpacking data into a dictionary for the firmware versions 2212 "skip"
    or "block". Uses the calibration data.

    Parameters
    ----------
    filename : str
        The name of the ".dat" data file.
    board_number : str
        The LinoSPAD2 dautherboard number.
    fw_ver : str
        2212 firmware version: either "skip" or "block".
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    dict
        Matrix of timestamps with 256 rows. Output is a dictionary as the number
        of columns is different for each row.

    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212b' or '2212s'")
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, 'NL11' or 'A5'")

    timestamp_list = {}

    for i in range(0, 256):
        timestamp_list["{}".format(i)] = []

    # Variables that follow the cycle and the TDC numbers
    cycler = 0
    tdc = 0

    # Function for assigning pixel addresses bases on the type of the 2212
    # firmware version
    def _pix_num(tdc_num, pix_coor):
        if fw_ver == "2212b":
            out = 4 * tdc_num + pix_coor
        elif fw_ver == "2212s":
            out = tdc_num + 64 * pix_coor

        return out

    with open(filename, "rb") as f:
        while True:
            rawpacket = f.read(4)
            # All steps are in units of 32 bits
            # Reaching the end of a cycle, assign a '-1' to each pixel
            if not cycler % (32 * 65 * timestamps) and cycler != 0:
                for i in range(256):
                    timestamp_list["{}".format(i)].append(-2)
            # Next TDC
            if not cycler % (32 * timestamps) and cycler != 0:
                tdc += 1
            cycler += 32
            # Cut the 64th TDC (TDC=[0,...63]) that does not contain timestamps
            if tdc != 0 and tdc == 64:
                continue
            # Reset the TDC number - end of cycle
            if tdc != 0 and tdc == 65:
                tdc = 0
            if not rawpacket:
                break
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:
                pix_coor = (packet[0] >> 28) & 0x3
                address = _pix_num(tdc, pix_coor)
                timestamp_list["{}".format(address)].append((packet[0] & 0xFFFFFFF))

    for key in timestamp_list:
        timestamp_list[key] = np.array(timestamp_list[key]).astype(np.longlong)

    # path to the current script, two levels up (the script itself is in the path) and
    # one level down to the calibration data
    path_calib_data = os.path.realpath(__file__) + "/../.." + "/calibration_data"

    try:
        cal_mat = calibrate_load(path_calib_data, board_number)
    except FileNotFoundError:
        print(
            "No .csv file with the calibration data was found, check the path "
            "or run the calibration."
        )
        sys.exit()
    for i in range(256):
        ind = np.where(np.array(timestamp_list["{}".format(i)]) >= 0)[0]
        if not np.any(ind):
            continue
        timestamp_list["{}".format(i)][ind] = (
            timestamp_list["{}".format(i)][ind]
            - timestamp_list["{}".format(i)][ind] % 140
        ) * 17.857 + cal_mat[i, (timestamp_list["{}".format(i)][ind] % 140)]

    return timestamp_list


def unpack_2212_numpy(file, board_number: str, timestamps: int = 512):
    """Unpack data from firmware version 2212.

    Unpacks binary-encoded data from LinoSPAD2 firmware version 2212 block.
    Uses numpy to achieve best speed for unpacking. Data is returned as a 3d array
    where rows are TDC numbers, columns are the data, each cell contains pixel
    number in the TDC (from 0 to 3) and the timestamp recorded by that pixel.

    Parameters
    ----------
    file : str
        '.dat' data file.
    board_number : str
        LinoSPAD2 daughterboard number. Either 'A5' or 'NL11' are recognized.
    timestamps : int, optional
        Number of timestamps per TDC per acquisition cycle. The default is 512.

    Raises
    ------
    TypeError
        Controller for the type of 'board_number' parameter which should be string.
    FileNotFoundError
        Controller for stopping the script in the case no calibration data
        were found.

    Returns
    -------
    data_all : array-like
        3d array of pixel coordinates in the TDC and the timestamps.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, 'NL11' or 'A5'")

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
        np.linspace(timestamps, cycles * timestamps, cycles).astype(np.longlong),
        -2,
        1,
    )

    data_matrix_t = np.insert(
        data_matrix_t,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(np.longlong),
        -2,
        1,
    )
    # combine both matrices into a single one, where each cell holds pix coordinates
    # in the TDC and the timestamp
    data_all = np.stack((data_matrix_p, data_matrix_t), axis=2).astype(np.longlong)

    # path to the current script, two levels up (the script itself is in the path)
    # and one level down to the calibration data
    pix_coor = np.arange(256).reshape(64, 4)
    path_calib_data = os.path.realpath(__file__) + "/../.." + "/calibration_data"

    try:
        cal_mat = calibrate_load(path_calib_data, board_number)
    except FileNotFoundError:
        raise FileNotFoundError(
            "No .csv file with the calibration data was found, check the path "
            "or run the calibration."
        )

    for i in range(256):
        # transform pixel number to TDC number and pixel coordinates in that TDC
        # (from 0 to 3)
        tdc, pix = np.argwhere(pix_coor == i)[0]
        # find data from that pixel
        ind = np.where(data_all[tdc].T[0] == pix)[0]
        # cut non-valid timestamps ('-1's)
        ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
        if not np.any(ind):
            continue
        # apply calibration
        data_all[tdc].T[1][ind] = (
            data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140
        ) * 17.857 + cal_mat[i, (data_all[tdc].T[1][ind] % 140)]

    return data_all
