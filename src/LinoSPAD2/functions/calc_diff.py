"""Module for computing timestamp differences.

Compares all timestamps from the same cycle for the given pair of pixels
against a given value (delta_window). Differences in that window are saved
and returned as a list.

This file can also be imported as a module and contains the following
functions:

    * calculate_differences_2212 - calculate timestamp differences for
    the given pair of pixels. Works only with firmware version '2212'.

"""

import numpy as np


def calculate_differences_2212(
    data1, data2, cycle_ends, delta_window: float = 10e3
):
    """Calculate timestamp differences for firmware 2212.

    Calculate timestamp differences for two given pixels and LinoSPAD2
    firmware version 2212.

    Parameters
    ----------
    data1 : array-like
        Array of data from the first pixel of the pair for which
        timestamp differences are calculated.
    data2 : array-like
        Array of data from the second pixel of the pair for which
        timestamp differences are calculated.
    cycle_ends : array-like
        Array of positions of '-2's that indicate ends of cycles.
    delta_window : float, optional
        A width of a window in which the number of timestamp differences
        are counted. The default value is 10 ns. The default is 10e3.

    Returns
    -------
    deltas_out : list
        All timestamp differences found in a given time window for a
        given pair of pixels.

    """
    deltas_out = []

    for cyc in range(len(cycle_ends) - 1):
        data1_cycle = data1[cycle_ends[cyc] : cycle_ends[cyc + 1]]
        data2_cycle = data2[cycle_ends[cyc] : cycle_ends[cyc + 1]]

        # calculate delta t
        tmsp1 = data1_cycle[np.where(data1_cycle > 0)[0]]
        tmsp2 = data2_cycle[np.where(data2_cycle > 0)[0]]
        for t1 in tmsp1:
            # calculate all timestamp differences in the given cycle
            deltas = tmsp2 - t1
            # take values in the given delta t window only
            ind = np.where(np.abs(deltas) < delta_window)[0]
            deltas_out.extend(deltas[ind])

    return deltas_out
