"""Module that contains functions cut from the 'functions' as these
are no longer utilized, only for debugging.


    * calc_diff_2208 - calculate timestamp differences for the given
    pair of pixels. Works only with firmware version '2208'.

"""

import numpy as np


def calc_diff_2208(
    data_pair, timestamps: int = 512, delta_window: float = 10e3
):
    """Calculate timestamp differences for firmware 2208.

    Calculate and return timestamp differences for the given pair of
    pixels (indicated by the data_pair - data from a single pair of
    pixels). Works only with the 2208 firmware version.

    Parameters
    ----------
    data_pair : array-like
        Data from 2 pixels.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition window. The
        default is 512.
    delta_window : float, optional
        A width of a window in which the number of timestamp differences
        are counted. The default value is 10 ns. The default is 10e3.

    Returns
    -------
    output : array-like
        An array of timestamp differences for the given pair of pixels.

    """
    timestamps_total = len(data_pair[0])

    output = []

    acq = 0  # number of acq cycle
    for j in range(timestamps_total):
        if j % timestamps == 0:
            acq = acq + 1  # next acq cycle
        if data_pair[0][j] == -1:
            continue
        for p in range(timestamps):
            n = timestamps * (acq - 1) + p
            if data_pair[1][n] == -1:
                continue
            else:
                delta = data_pair[0][j] - data_pair[1][n]
                if np.abs(delta) > delta_window:
                    continue
                else:
                    output.append(delta)
    return output
