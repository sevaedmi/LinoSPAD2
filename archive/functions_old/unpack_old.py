"""Module that contains functions cut from the 'functions' as these
are no longer utilized, only for debugging.

Following functions can be found in this module.

    * unpack_binary_flex

    * unpack_mult_cut
"""


def unpack_binary_flex(filename, timestamps: int = 512):
    """Unpack binary data from the LinoSPAD2.

    Unpacking a single 'dat' output of the LinoSPAD2. Due to the
    straightforward approach used mainly for debugging, otherwise is
    pretty slow.

    Parameters
    ----------
    filename : str
        File with data from LinoSPAD2 in which precisely timestamps
        lines of data per acquistion cycle is written.
    timestamps: int, optional
        Number of binary-encoded timestamps in the 'dat' file. The
        default value is 512.

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


def unpack_mult_cut(files, pixels, board_number: str, timestamps: int = 512):
    """Unpack binary data from LinoSPAD2 only for given pixels.

    Returns timestamps only for the given pixels. Uses the calibration data.

    Parameters
    ----------
    files : list
        List of files' names with the binary data from LinoSPAD2.
    pixels : array-like or list
        Array or list of pixel numbers for which the data should be unpacked.
    board_number : str
        The LinoSPAD2 daughterboard number.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    ndarray-like
        A matrix of pixels X timestamps*number_of_cycles of timestamps.

    """
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
