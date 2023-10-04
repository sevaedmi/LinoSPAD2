"""Module for working with the calibration data.

This file can also be imported as a module and contains the following
functions:

    * calibrate_save - calculate a calibration matrix and save it as
    a '.csv' table.

    * calibrate_load - load the calibration matrix from a '.csv' table.
"""
import glob
import os

import numpy as np
import pandas as pd


def calibrate_save(path, timestamps: int = 512):
    """Calculate and save calibration data.

    Function for calculating the calibration matrix and saving it into a
    '.csv' file. The data file used for the calculation should be taken
    with the sensor uniformly illuminated by ambient light.

    Parameters
    ----------
    path : str
        Path to the data file.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The
        default is 512.

    Returns
    -------
    None.

    """
    os.chdir(path)
    filename = glob.glob("*.dat*")[0]

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
    cal_mat_df.to_csv("Calibration_data.csv")


def calibrate_load(path, board_number: str, fw_ver: str):
    """Load the calibration data.

    Parameters
    ----------
    path : str
        Path to the '.csv' file with the calibration matrix.
    board_number: str
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
    file_TDC = glob.glob("*TDC_{bn}_{fw}*".format(bn=board_number, fw=fw_ver))[
        0
    ]
    # Compensating for offset
    file_offset = glob.glob(
        "*Offset_{bn}_{fw}*".format(bn=board_number, fw=fw_ver)
    )[0]

    # Skipping first row of TDC bins' numbers
    data_matrix_TDC = np.genfromtxt(file_TDC, delimiter=",", skip_header=1)
    # Cut the first column which is pixel numbers
    data_matrix_TDC = np.delete(data_matrix_TDC, 0, axis=1)

    offset_arr = np.load(file_offset)

    os.chdir(path_to_back)

    return data_matrix_TDC, offset_arr
