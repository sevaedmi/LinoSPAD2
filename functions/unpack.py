"""Unpack data from LinoSPAD2

Functions for unpacking either 'txt' of 'dat' data files of LinoSPAD2.
Functions for either 10, 512 or a given number of timestamps per acquisition
cycle per pixel are available.

This file can also be imported as a module and contains the following
functions:

    * unpack_txt_512 - unpacks the 'txt' data files with 512 timestamps
    per acquisition cycle

    * unpack_txt_10 - unpacks the 'txt' data files with 10 timestamps per
    acquisition cycle

    * unpack_binary_10 - unpacks the 'dat' data files with 10 timestamps
    per acquisition cycle

    * unpack_binary_512 - unpack the 'dat' data files with 512 timestamps
    per acquisition point

    * unpack_binary_flex - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle

    * unpack_binary_df - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle into a pandas dataframe. Unlike the
    others, this functions does not write '-1' nonvalid timestamps which makes
    this the fastest approach compared to other. The dataframe output allows
    faster plots using seaborn.

    * unpack_calib - unpacks the 'dat' data files with a given number of
    timestamps per acquisition cycle. Uses the calibration data. Imputing the
    LinoSPAD2 board number is required.

"""

from struct import unpack
import numpy as np
import pandas as pd
from functions.calibrate import calibrate_load
import sys
import os
from glob import glob


def unpack_txt_512(filename):
    """Unpacks the 'txt' data files with 512 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 512 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 512*number-of-cycles) of data points.

    """

    data = np.genfromtxt(filename)
    data_matrix = np.zeros((256, int(len(data) / 256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(data) / 512 / 256  # number of cycles, 512 data lines per pixel per
    # cycle, 256 pixels

    # =========================================================================
    # Unpack data from the txt, which is a Nx1 array, into a 256x#cycles matrix
    # using two counters. When i (number of the pixel) reaches the 255
    # (256th pixel), move k one step further - second data acquisition cycle,
    # when the algorithm goes back to the 1st pixel and writes the data right
    # next to the data from previous cycle.
    # =========================================================================
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 512 : k * 512 + 512] = (
                data[(i + 256 * k) * 512 : (i + 256 * k) * 512 + 512] - 2**31
            )
            i = i + 1
        k = k + 1
    data_matrix = data_matrix * 17.857  # 2.5 ns from TDC 400 MHz clock read
    # out 140 bins from 35 elements of the delay line - average bin size
    # is 17.857 ps

    return data_matrix


def unpack_txt_10(filename):
    """Unpacks the 'txt' data files with 10 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 10 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 10*number-of-cycles) of data points.

    """

    data = np.genfromtxt(filename)
    data_matrix = np.zeros((256, int(len(data) / 256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(data) / 10 / 256  # number of cycles, 10 data lines per pixel per
    # cycle, 256 pixels

    # =====================================================================
    # Unpack data from the txt, which is a Nx1 array, into a 256x#cycles
    # matrix using two counters. When i (number of the pixel) reaches the
    # 255 (256th pixel), move k one step further - second data acquisition
    # cycle, when the algorithm goes back to the 1st pixel and writes the
    # data right next to the data from previous cycle.
    # =====================================================================
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 10 : k * 10 + 10] = (
                data[(i + 256 * k) * 10 : (i + 256 * k) * 10 + 10] - 2**31
            )
            i = i + 1
        k = k + 1
    data_matrix = data_matrix * 17.857  # 2.5 ns from TDC 400 MHz clock read
    # out 140 bins from 35 elements of the delay line - average bin size
    # is 17.857 ps

    return data_matrix


def unpack_binary_10(filename):
    """Unpacks the 'dat' data files with 10 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 10 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 10*number-of-cycles) of data points.

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
                # elements of the delay line - average bin sizу is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
            # not in this firmware??
            address_list.append(address)
    # rows=#pixels, cols=#cycles
    data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))

    noc = len(timestamp_list) / 10 / 256  # number of cycles, 10 data lines per
    # pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 10 : k * 10 + 10] = timestamp_list[
                (i + 256 * k) * 10 : (i + 256 * k) * 10 + 10
            ]
            i = i + 1
        k = k + 1
    return data_matrix


def unpack_binary_512(filename):
    """Unpacks the 'dat' data files with 512 timestamps per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 512 timestamps
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 512*number-of-cycles) of data points.

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
                # elements of the delay line - average bin sizу is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
            # not in this firmware??
            address_list.append(address)
    # rows=#pixels, cols=#cycles
    data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))

    noc = len(timestamp_list) / 512 / 256  # number of cycles, 512 data lines per
    # pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k * 512 : k * 512 + 512] = timestamp_list[
                (i + 256 * k) * 512 : (i + 256 * k) * 512 + 512
            ]
            i = i + 1
        k = k + 1
    return data_matrix


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


def unpack_binary_df(
    filename, timestamps: int = 512, apply_mask: bool = True, cut_empty: bool = True
):
    """Unpacks the 'dat' files with a given number of timestamps per acquisition cycle
    per pixel into a pandas dataframe. The fastest unpacking compared to others.

    Parameters
    ----------
    filename : str
        The 'dat' binary-encoded datafile.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.
    apply_mask : bool, optional
        Switch for masking the warm/hot pixels. Default is True.
    cut_empty : bool, optional
        Switch for appending '-1', or either non-valid or empty timestamps, to the
        output. Default is True.

    Returns
    -------
    timestamps : pandas.DataFrame
        A dataframe with two columns: first is the pixel number, second are the
        timestamps.

    """

    mask = [
        15,
        16,
        29,
        39,
        40,
        50,
        52,
        66,
        73,
        93,
        95,
        96,
        98,
        101,
        109,
        122,
        127,
        196,
        210,
        231,
        236,
        238,
    ]

    timestamp_list = list()
    pixels_list = list()
    cycles_list = list()
    acq = 1
    i = 0
    cycles = timestamps * 256

    with open(filename, "rb") as f:
        while True:
            i += 1
            rawpacket = f.read(4)
            if not rawpacket:
                break
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:
                # timestamps cover last 28 bits of the total 32
                timestamp = packet[0] & 0xFFFFFFF
                # 17.857 ps - average bin size
                timestamp = timestamp * 17.857
                pixels_list.append(int(i / 512) + 1)
                timestamp_list.append(timestamp)
                cycles_list.append(acq)
            elif cut_empty is False:
                timestamp_list.append(-1)
                pixels_list.append(int(i / 512 + 1))
                cycles_list.append(acq)
            if i == cycles:
                i = 0
                acq += 1
    dic = {"Pixel": pixels_list, "Timestamp": timestamp_list, "Cycle": cycles_list}
    timestamps = pd.DataFrame(dic)
    timestamps = timestamps[~timestamps["Pixel"].isin(mask)]

    return timestamps


# def unpack_numpy(filename, timestamps: int = 512):
#     """
#     Function for unpacking binary data based on the numpy library.

#     Parameters
#     ----------
#     filename : str
#         Name of the file with the binary-encoded data.
#     timestamps : int, optional
#         Number of timestamps per acquisition cycle per pixel. Default is 512.

#     Returns
#     -------
#     data_matrix : array_like
#         A 2D matrix (256 pixels by timestamps X number-of-cycles) of
#         timestamps.

#     """
#     rawFile = np.fromfile(filename, dtype=np.uint32)  # read data
#     data = (rawFile & 0xFFFFFFF).astype(int) * 17.857  # Multiply with the lowes bin
#     data[np.where(rawFile < 0x80000000)] = -1  # Mask not valid data
#     cycles = int(len(data) / timestamps / 256)  # number of cycles,
#     data_matrix = (
#         data.reshape((timestamps, cycles * 256), order="F")
#         .reshape((timestamps, 256, -1), order="F")
#         .transpose((0, 2, 1))
#         .reshape((-1, 256), order="F")
#         .transpose()
#     )  # reshape the matrix
#     return data_matrix


def unpack_numpy(filename, timestamps: int = 512):
    """
    Function for unpacking the binary-encoded data from the LinoSPAD2
    based on the numpy library. Currently, the fastest option for
    unpacking.

    Parameters
    ----------
    filename : str
        Name of the file with the data.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
        The default is 512.

    Returns
    -------
    output : ndarray
        A 2D matrix (256 x timestamps*number_of_cycles) of timestamps.

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
    # lowest 28 bits are the timestamp; convert to ps
    data = (rawFile & 0xFFFFFFF).astype(np.longlong) * 17.857
    # mask nonvalid data with '-1'
    data[np.where(rawFile < 0x80000000)] = -1
    # number of acquisition cycles
    cycles = int(len(data) / timestamps / 256)

    data_matrix = (
        data.reshape(cycles, 256, timestamps)
        .transpose((1, 0, 2))
        .reshape(256, timestamps * cycles)
    )

    return data_matrix


def unpack_calib(filename, board_number: str, timestamps: int = 512):
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


def unpack_calib_mult(path, board_number: str, timestamps: int = 512):

    os.chdir(path)

    files = glob("*.dat*")

    data_all = []

    for i, file in enumerate(files):
        if not np.any(data_all):
            data_all = unpack_calib(file, board_number, timestamps)
        else:
            data_all = np.append(data_all, unpack_numpy(file, timestamps), axis=1)

    return data_all
