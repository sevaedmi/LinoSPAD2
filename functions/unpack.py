"""Unpack data from LinoSPAD2

Functions for unpacking either 'txt' of 'dat' data files of LinoSPAD2.
Functions for both 10 and 512 lines of data per acquisition cycle are
available.

Works with both 'txt' and 'dat' files.

This file can also be imported as a module and contains the following
functions:

    * unpack_txt_512 - unpacks the 'txt' data files with 512 lines of data
    per acquisition cycle
    * unpack_txt_10 - unpacks the 'txt' data files with 10 lines of data per
    acquisition cycle
    * unpack_binary_10 - unpacks the 'dat' data files with 10 lines of data
    per acquistion cycle
    
    # TODO: flexible function for unpacking N lines of binary-encoded data

"""

from struct import unpack
import numpy as np


def unpack_txt_512(filename):
    """Unpacks the 'txt' data files with 512 lines of data per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 512 lines of data
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 512*number-of-cycles) of data points.

    """

    data = np.genfromtxt(filename)
    data_matrix = np.zeros((256, int(len(data)/256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(data)/512/256  # number of cycles, 512 data lines per pixel per
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
            data_matrix[i][k*512:k*512+512] = data[(i+256*k)*512:
                                                   (i+256*k)*512+512]-2**31
            i = i+1
        k = k+1

    data_matrix = data_matrix*17.857  # 2.5 ns from TDC 400 MHz clock read
    # out 140 bins from 35 elements of the delay line - average bin size
    # is 17.857 ps

    # Cut the nonscence and insert -1 where there is no timestamp
    for i, num in enumerate(data_matrix):
        data_matrix[i][np.where(num < 0)[0]] = -1

    return data_matrix


def unpack_txt_10(filename):
    """Unpacks the 'txt' data files with 10 lines of data per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 10 lines of data
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 10*number-of-cycles) of data points.

    """

    data = np.genfromtxt(filename)
    data_matrix = np.zeros((256, int(len(data)/256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(data)/10/256  # number of cycles, 10 data lines per pixel per
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
            data_matrix[i][k*10:k*10+10] = data[(i+256*k)*10:
                                                (i+256*k)*10+10]-2**31
            i = i+1
        k = k+1

    data_matrix = data_matrix*17.857  # 2.5 ns from TDC 400 MHz clock read
    # out 140 bins from 35 elements of the delay line - average bin size
    # is 17.857 ps

    # Cut the nonscence and insert -1 where there is no timestamp
    for i, num in enumerate(data_matrix):
        data_matrix[i][np.where(num < 0)[0]] = -1

    return data_matrix


def unpack_binary_10(filename):
    """Unpacks the 'dat' data files with 10 lines of data per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 10 lines of data
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 10*number-of-cycles) of data points.

    """

    timestamp_list = []
    address_list = []

    with open(filename, 'rb') as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack('<I', rawpacket)
            if (packet[0] >> 31) == 1:  # check validity bit: if 1
                # - timestamp is valid
                timestamp = packet[0] & 0xfffffff  # cut the higher bits,
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
    data_matrix = np.zeros((256, int(len(timestamp_list)/256)))

    noc = len(timestamp_list)/10/256  # number of cycles, 10 data lines per
    # pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k*10:k*10+10] = timestamp_list[(i+256*k)*10:
                                                          (i+256*k)*10+10]
            i = i+1
        k = k+1

    # Cut the nonscence and insert -1 where there is no timestamp
    for i, num in enumerate(data_matrix):
        data_matrix[i][np.where(num < 0)[0]] = -1

    return data_matrix


def unpack_binary_512(filename):
    """Unpacks the 'dat' data files with 512 lines of data per acquistion
    cycle.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely 512 lines of data
        per acquistion cycle is written.

    Returns
    -------
    data_matrix : array_like
        2D matrix (256 pixels by 512*number-of-cycles) of data points.

    """

    timestamp_list = []
    address_list = []

    with open(filename, 'rb') as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack('<I', rawpacket)
            if (packet[0] >> 31) == 1:  # check validity bit: if 1
                # - timestamp is valid
                timestamp = packet[0] & 0xfffffff  # cut the higher bits,
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
    data_matrix = np.zeros((256, int(len(timestamp_list)/256)))

    noc = len(timestamp_list)/512/256  # number of cycles, 512 data lines per
    # pixel per cycle, 256 pixels

    # pack the data from a 1D array into a 2D matrix
    k = 0
    while k != noc:
        i = 0
        while i < 256:
            data_matrix[i][k*512:k*512+512] = timestamp_list[(i+256*k)*512:
                                                             (i+256*k)*512+512]
            i = i+1
        k = k+1

    # Cut the nonscence and insert -1 where there is no timestamp
    for i, num in enumerate(data_matrix):
        data_matrix[i][np.where(num < 0)[0]] = -1

    return data_matrix
