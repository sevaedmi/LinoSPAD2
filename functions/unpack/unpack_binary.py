""" Unpack and decode the binary data output of the LinoSPAD2."""

import numpy as np
from struct import unpack


def unpack(filename):
    timestamp_list = []
    address_list = []

    with open(filename, 'rb') as f:
        while True:
            rawpacket = f.read(4)  # read 32 bits
            if not rawpacket:
                break  # stop when the are no further 4 bytes to readout
            packet = unpack('<I', rawpacket)
            # if (packet[0] >> 31) == 1:  # check validity bit: if 1
            # - timestamp is valid
            timestamp = packet[0] & 0xfffffff  # cut the higher bits, leave
            # only timestamp ones
            timestamp_list.append(timestamp)
            address = (packet[0] >> 28) & 0x3  # gives away only zeroes - not in
            # this firmware??
            address_list.append(address)

    timestamp_list = np.array(timestamp_list)*17.857  # in ps
    timestamps = np.delete(timestamp_list, np.where(timestamp_list == 0)[0])

    return timestamps
