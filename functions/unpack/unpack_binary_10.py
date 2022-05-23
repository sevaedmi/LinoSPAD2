""" Unpack and decode the binary data output of the LinoSPAD2.
For files with 10 data lines per pixel per acqusition cycle.
"""

import numpy as np
from struct import unpack


def unpack_binary(filename):
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
                timestamp = packet[0] & 0xfffffff  # cut the higher bits, leave
                # only timestamp ones
                # 2.5 ns from TDC 400 MHz clock read out 140 bins from 35
                # elements of the delay line - average bin sizу is 17.857 ps
                timestamp = timestamp * 17.857  # in ps
            else:
                timestamp = -1
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes - not
            # in this firmware??
            address_list.append(address)

    Data_matrix = np.zeros((256, int(len(timestamp_list)/256)))  # rows=#pixels,
    # cols=#cycles

    noc = len(timestamp_list)/10/256  # number of cycles, 10 data lines per
    # pixel per cycle, 256 pixels

    k = 0
    while k != noc:
        i = 0
        while i < 256:
            Data_matrix[i][k*10:k*10+10] = timestamp_list[(i+256*k)*10:
                                                          (i+256*k)*10+10]
            i = i+1
        k = k+1

    # Cut the nonscence and insert -1 where there is no timestamp
    for i in range(len(Data_matrix)):
        Data_matrix[i][np.where(Data_matrix[i] < 0)[0]] = -1

    return Data_matrix
