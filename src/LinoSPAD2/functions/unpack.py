"""Module with scripts for unpacking data from LinoSPAD2.

This file can also be imported as a module and contains the following
functions:

    * unpack_bin - function for unpacking data from LinoSPAD2,
    firmware version 2212. Utilizes the numpy library to speed up the
    process.

"""

import os

import numpy as np

from LinoSPAD2.functions.calibrate import calibrate_load


def unpack_bin(file, board_number: str, fw_ver: str, timestamps: int = 512):
    """Unpack data from firmware version 2212.

    Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.
    Uses numpy to achieve the best speed for unpacking. Data is returned
    as a 3D array where rows are TDC numbers, columns are the data, and
    each cell contains pixel number in the TDC (from 0 to 3) and the
    timestamp recorded by that pixel.

    Parameters
    ----------
    file : str
        '.dat' data file.
    board_number : str
        LinoSPAD2 daughterboard number. Either 'A5' or 'NL11' are
        recognized.
    fw_ver : str
        LinoSPAD2 firmware version. Either '2212s' or '2212b' are accepted.
    timestamps : int, optional
        Number of timestamps per TDC per acquisition cycle. The default
        is 512.

    Raises
    ------
    TypeError
        Controller for the type of 'board_number' parameter which should
        be a string. FileNotFoundError
        Controller for stopping the script in the case no calibration
        data were found.

    Returns
    -------
    data_all : array-like
        3D array of pixel coordinates in the TDC and the timestamps.

    """
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, 'NL11' or 'A5'")
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

    # path to the current script, two levels up (the script itself is in
    # the path) and two levels down to the calibration data
    pix_coor = np.arange(256).reshape(64, 4)
    path_calib_data = (
        os.path.realpath(__file__) + "/../.." + "/params/calibration_data"
    )

    try:
        cal_mat, offset_arr = calibrate_load(
            path_calib_data, board_number, fw_ver
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "No .csv file with the calibration data was found, check the path "
            "or run the calibration."
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
        # apply calibration; offset is added due to how delta ts are
        # calculated
        data_all[tdc].T[1][ind] = (
            (data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140) * 17.857
            + cal_mat[i, (data_all[tdc].T[1][ind] % 140)]
            + offset_arr[i]
        )

    return data_all
