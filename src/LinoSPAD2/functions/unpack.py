"""Module with scripts for unpacking data from LinoSPAD2.

This file can also be imported as a module and contains the following
functions:

    * unpack_bin - function for unpacking data from LinoSPAD2,
    firmware version 2212. Utilizes the numpy library to speed up the
    process.
    
    * unpack_bin_wat - function for unpacking data from LinoSPAD2,
    including the absolute timestamps, works with firmware version 2212.

"""

import os

import numpy as np

from LinoSPAD2.functions.calibrate import load_calibration_data


def unpack_binary_data(
    file: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    include_offset: bool = True,
    apply_calibration: bool = True,
):
    """Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.

    Parameters
    ----------
    file : str
        Path to the binary data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b' (block).
    timestamps : int, optional
        Number of timestamps per cycle per TDC per acquisition cycle.
        The default is 512.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC
        calibration is applied. The default is True.

    Returns
    -------
    data_all : array-like
        3D array of pixel coordinates in the TDC and the timestamps.

    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or 'firmware_version'
        parameters are not of string type.
    FileNotFoundError
        If no calibration data file is found.

    Notes
    -----
    The returned data is a 3D array where rows represent TDC numbers,
    columns represent the data, and each cell contains a pixel number in
    the TDC (from 0 to 3) and the timestamp recorded by that pixel.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str):
        raise TypeError("'firmware_version' should be a string.")

    # Unpack binary data
    raw_data = np.fromfile(file, dtype=np.uint32)
    # Timestamps are stored in the lower 28 bits
    data_timestamps = (raw_data & 0xFFFFFFF).astype(np.longlong)
    # Pixel address in the given TDC is 2 bits above timestamp
    data_pixels = ((raw_data >> 28) & 0x3).astype(np.longlong)
    data_timestamps[np.where(raw_data < 0x80000000)] = -1
    # Number of acquisition cycles in each data file
    cycles = int(len(data_timestamps) / timestamps / 65)
    # Transform into a matrix of size 65 by cycles*timestamps
    data_matrix_pixels = (
        data_pixels.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )

    data_matrix_timestamps = (
        data_timestamps.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )
    # Cut the 65th TDC that does not hold any actual data from pixels
    data_matrix_pixels = data_matrix_pixels[:-1]
    data_matrix_timestamps = data_matrix_timestamps[:-1]
    # Insert '-2' at the end of each cycle
    data_matrix_pixels = np.insert(
        data_matrix_pixels,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )

    data_matrix_timestamps = np.insert(
        data_matrix_timestamps,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )
    # Combine both matrices into a single one, where each cell holds pixel
    # coordinates in the TDC and the timestamp
    data_all = np.stack(
        (data_matrix_pixels, data_matrix_timestamps), axis=2
    ).astype(np.longlong)

    if apply_calibration is False:
        data_all[:, :, 1] = data_all[:, :, 1] * 17.857
    else:
        # Path to the calibration data
        pix_coordinates = np.arange(256).reshape(64, 4)
        # path_calibration_data = (
        #     os.path.realpath(__file__) + "/../.." + "/params/calibration_data"
        # )

        path_calibration_data = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "params",
            "calibration_data",
        )

        try:
            if include_offset:
                calibration_matrix, offset_array = load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    include_offset,
                )
            else:
                calibration_matrix = load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    include_offset,
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                "No .csv file with the calibration data was found. "
                "Check the path or run the calibration."
            )

        for i in range(256):
            # Transform pixel number to TDC number and pixel coordinates in
            # that TDC (from 0 to 3)
            tdc, pix = np.argwhere(pix_coordinates == i)[0]
            # Find data from that pixel
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            # Cut non-valid timestamps ('-1's)
            ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
            if not np.any(ind):
                continue
            # Apply calibration; offset is added due to how delta ts are
            # calculated
            if include_offset:
                data_all[tdc].T[1][ind] = (
                    (data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140)
                    * 17.857
                    + calibration_matrix[i, (data_all[tdc].T[1][ind] % 140)]
                    + offset_array[i]
                )
            else:
                data_all[tdc].T[1][ind] = (
                    data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140
                ) * 17.857 + calibration_matrix[
                    i, (data_all[tdc].T[1][ind] % 140)
                ]

    return data_all


def unpack_binary_data_with_absolute_timestamps(
    file_path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    include_offset: bool = True,
    apply_calibration: bool = True,
):
    """Unpacks binary-encoded data from LinoSPAD2 firmware version 2212
    with absolute timestamps.

    Parameters
    ----------
    file_path : str
        Path to the binary data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b'
        (block) are accepted.
    timestamps : int, optional
        Number of timestamps per TDC per acquisition cycle.
        The default is 512.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC
        calibration is applied. The default is True.

    Returns
    -------
    data_all : array-like
        3D array of pixel coordinates in the TDC and the timestamps.
    absolute_timestamps : array-like
        Array of absolute timestamps in ps, with length equal to the
        number of acquisition cycles, as the absolute timestamps are
        recorded at the start of each cycle.

    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or
        'firmware_version' parameters are not of string type.
    FileNotFoundError
        If no calibration data file is found.

    Notes
    -----
    The returned data is a 3D array where rows represent TDC numbers,
    columns represent the data, and each cell contains a pixel number in
    the TDC (from 0 to 3) and the timestamp recorded by that pixel.
    Additionally, an array of absolute timestamps is returned, representing
    the absolute time at the start of each acquisition cycle.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str):
        raise TypeError("'firmware_version' should be a string.")

    # Unpack binary data
    raw_data = np.fromfile(file_path, dtype=np.uint32)
    # Timestamps are stored in the lower 28 bits
    data_raw = (raw_data & 0xFFFFFFF).astype(np.longlong)
    cycles = int(len(data_raw) / (timestamps * 65 + 2))

    # List of indices of absolute timestamps
    ind = []
    # Converted absolute timestamps: from binary to ps
    absolute_timestamps = np.zeros(cycles)

    # Collect the absolute timestamps indices
    for cyc in range(cycles):
        ind.extend(
            [
                x
                for x in range(
                    cyc * (65 * timestamps + 2),
                    cyc * (65 * timestamps + 2) + 2,
                )
            ]
        )

    # Absolute timestamps only
    data_absolute_timestamps = data_raw[ind].reshape(cycles, 2)

    # Convert the absolute timestamps from binary to decimal (ps)
    for cyc in range(cycles):
        absolute_timestamps[cyc] = int(
            "0b"
            + bin(int(data_absolute_timestamps[cyc][1]))[2:]
            + bin(int(data_absolute_timestamps[cyc][0]))[2:],
            2,
        )

    # Standard data: everything besides the absolute timestamps
    data_standard = np.delete(data_raw, ind)

    # Pixel address in the given TDC is 2 bits above timestamp
    data_pixels = ((data_standard >> 28) & 0x3).astype(np.longlong)
    data_standard[np.where(data_standard < 0x80000000)] = -1
    # Number of acquisition cycles in each data file

    # Transform into matrix 65 by cycles*timestamps
    data_matrix_pixels = (
        data_pixels.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )

    data_matrix_timestamps = (
        data_standard.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, timestamps * cycles)
    )
    # Cut the 65th TDC that does not hold any actual data from pixels
    data_matrix_pixels = data_matrix_pixels[:-1]
    data_matrix_timestamps = data_matrix_timestamps[:-1]
    # Insert '-2' at the end of each cycle
    data_matrix_pixels = np.insert(
        data_matrix_pixels,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )

    data_matrix_timestamps = np.insert(
        data_matrix_timestamps,
        np.linspace(timestamps, cycles * timestamps, cycles).astype(
            np.longlong
        ),
        -2,
        1,
    )
    # Combine both matrices into a single one, where each cell holds pixel
    # coordinates in the TDC and the timestamp
    data_all = np.stack(
        (data_matrix_pixels, data_matrix_timestamps), axis=2
    ).astype(np.longlong)

    if apply_calibration is False:
        data_all[:, :, 1] = data_all[:, :, 1] * 17.857
    else:
        # Path to the calibration data
        pixel_coordinates = np.arange(256).reshape(64, 4)
        # path_calibration_data = (
        #     os.path.realpath(__file__) + "/../.." + "/params/calibration_data"
        # )
        path_calibration_data = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "params",
            "calibration_data",
        )

        try:
            if include_offset:
                calibration_matrix, offset_array = load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    include_offset,
                )
            else:
                calibration_matrix = load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    include_offset,
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                "No .csv file with the calibration data was found. "
                "Check the path or run the calibration."
            )

        for i in range(256):
            # Transform pixel number to TDC number and pixel coordinates in
            # that TDC (from 0 to 3)
            tdc, pix = np.argwhere(pixel_coordinates == i)[0]
            # Find data from that pixel
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            # Cut non-valid timestamps ('-1's)
            ind = ind[np.where(data_all[tdc].T[1][ind] >= 0)[0]]
            if not np.any(ind):
                continue
            # Apply calibration; offset is added due to how delta ts are
            # calculated
            if include_offset:
                data_all[tdc].T[1][ind] = (
                    (data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140)
                    * 17.857
                    + calibration_matrix[i, (data_all[tdc].T[1][ind] % 140)]
                    + offset_array[i]
                )
            else:
                data_all[tdc].T[1][ind] = (
                    data_all[tdc].T[1][ind] - data_all[tdc].T[1][ind] % 140
                ) * 17.857 + calibration_matrix[
                    i, (data_all[tdc].T[1][ind] % 140)
                ]

    return data_all, absolute_timestamps
