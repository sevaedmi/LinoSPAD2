"""Unpack data from LinoSPAD2

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

    * unpack_mult - unpacks all 'dat' files in the given directory.
    Takes the number of timestamps per pixel per acquisition cycle and
    the LinoSPAD2 board number for the appropriate calibration data as
    parameters. Utilizes the 'unpack_calib' function for unpacking the
    files.

    * unpack_mult_cut - unpacks all 'dat' files in the given directory.
    Returns data only for the requested pixels.

    * unpack_2212 - function for unpacking data into a dictionary for the
    LinoSPAD2 firmware 2212 versions "skip" and "block".


"""

from struct import unpack
import numpy as np
from functions.calibrate import calibrate_load
import sys
import os
from glob import glob


def unpack_binary_flex(filename, timestamps: int = 512):
    """Unpacks the 'dat' data files with certain timestamps per acquistion
    cycle.

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


def unpack_numpy(filename, board_number: str, timestamps: int = 512):
    """
    Function for unpacking the .dat data files using the calibration
    data. The output is a matrix of '256 x timestamps*number_of_cycles'
    timestamps in ps.

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
    return data_matrix


def unpack_mult(path, board_number: str, timestamps: int = 512):
    """
    Function for unpacking all .dat data files in the directory using
    the calibration data. The output is a matrix of
    '256 x timestamps*number_of_cycles' timestamps in ps.

    Parameters
    ----------
    path : str
        Path to the data files.
    board_number : str
        LinoSPAD2 board number.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle.
        The default is 512.

    Returns
    -------
    data_all : array-like
        Matrix of '256 x timestamps*number_of_cycles' timestamps in ps.
    files_names : str
        First and last files' names. Can be used in other functions
        for convenient plot naming.

    """

    os.chdir(path)

    files = glob("*.dat*")

    files_names = files[0][:-4] + "-" + files[-1][:-4]

    data_all = []

    for i, file in enumerate(files):
        if not np.any(data_all):
            data_all = unpack_numpy(file, board_number, timestamps)
        else:
            data_all = np.append(
                data_all, unpack_numpy(file, board_number, timestamps), axis=1
            )

    return data_all, files_names


# def unpack_calib_mult_cut(path, pixels, board_number: str, timestamps: int = 512):
#     pixels = np.sort(pixels)

#     os.chdir(path)

#     files = glob("*.dat*")

#     files_names = files[0][:-4] + "-" + files[-1][:-4]

#     data_all = []

#     for i, file in enumerate(files):
#         if not np.any(data_all):
#             data_all = unpack_calib(file, board_number, timestamps)
#         else:
#             data_all = np.append(data_all, unpack_numpy(file, timestamps), axis=1)
#     output = []
#     for i in range(len(pixels)):
#         if not np.any(output):
#             output = data_all[pixels[0]]
#         else:
#             output = np.vstack((output, data_all[pixels[i]]))

#     return output, files_names


def unpack_mult_cut(files, pixels, board_number: str, timestamps: int = 512):
    pixels = np.sort(pixels)

    data_all = []

    for i, file in enumerate(files):
        if not np.any(data_all):
            data_all = unpack_numpy(file, board_number, timestamps)
        else:
            data_all = np.append(
                data_all, unpack_numpy(file, board_number, timestamps), axis=1
            )

    output = []

    for i in range(len(pixels)):
        if not np.any(output):
            output = data_all[pixels[0]]
        else:
            output = np.vstack((output, data_all[pixels[i]]))

    return output


def unpack_2212(filename, board_number, fw_ver, timestamps: int = 512):
    '''
    Function for unpacking data into a dictionary for the firmware versions 2212 "skip"
    and "block". Uses the calibration data.

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
    TYPE
        DESCRIPTION.

    '''
    rawFile = np.fromfile(filename, dtype=np.uint32)
    address = ((rawFile >> 28) & 0x3).astype(np.int64)
    data = (rawFile & 0xFFFFFFF).astype(np.longlong)
    data[np.where(rawFile < 0x80000000)] = -1

    def _pix_num(tdc_num, pix_coor):
        if fw_ver == "block":
            out = 4 * tdc_num + pix_coor
        elif fw_ver == "skip":
            out = tdc_num + 64 * pix_coor

        return out

    output = {}

    for i in range(0, 256):
        output["{}".format(i)] = []
    tdc_num = 0

    for i in range(0, len(data)):
        if i != 0 and i % timestamps == 0:
            tdc_num += 1
        if tdc_num != 0 and tdc_num == 64:
            continue
        if tdc_num != 0 and tdc_num == 65:
            tdc_num = 0
        pix_add = _pix_num(tdc_num, address[i])
        output["{}".format(pix_add)].append(data[i])

    # Calibration
    path_calib_data = os.path.realpath(__file__) + "/../.." + "/calibration_data"

    try:
        cal_mat = calibrate_load(path_calib_data, board_number)
    except FileNotFoundError:
        print(
            "No .csv file with the calibration data was found, check the path "
            "or run the calibration."
        )
        sys.exit()

    for i in range(0, 256):
        a = np.array(output["{}".format(i)])
        if not any(a):
            continue
        ind = np.where(a >= 0)[0]
        a[ind] = (a[ind] - a[ind] % 140) * 17.857 + cal_mat[i, (a[ind] % 140)]

    return output
