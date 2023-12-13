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
    data, pixels, pix_coor, delta_window: float = 50e3
):
    """Calculate timestamp differences for firmware version 2212.

    Calculate timestamp differences for the given pixels and LinoSPAD2
    firmware version 2212.

    Parameters
    ----------
    data : list of array-like
        List of data arrays, each corresponding to a different TDC.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated or list of two lists with pixel numbers for peak
        vs. peak calculations.
    pix_coor : array-like
        Array for transforming pixel number into TDC number + pixel
        coordinates.
    delta_window : float, optional
        Width of the time window for counting timestamp differences.
        The default is 50e3 (50 ns).

    Returns
    -------
    deltas_all : dict
        Dictionary containing timestamp differences for each pair of pixels.

    """
    deltas_all = {}

    # Most probably wrong indexing
    # for cyc in range(len(cycle_ends) - 1):
    #     data1_cycle = data1[cycle_ends[cyc] : cycle_ends[cyc + 1]]
    #     data2_cycle = data2[cycle_ends[cyc] : cycle_ends[cyc + 1]]

    #     # calculate delta t
    #     tmsp1 = data1_cycle[np.where(data1_cycle > 0)[0]]
    #     tmsp2 = data2_cycle[np.where(data2_cycle > 0)[0]]
    #     for t1 in tmsp1:
    #         # calculate all timestamp differences in the given cycle
    #         deltas = tmsp2 - t1
    #         # take values in the given delta t window only
    #         ind = np.where(np.abs(deltas) < delta_window)[0]
    #         deltas_out.extend(deltas[ind])

    if isinstance(pixels[0], list) and isinstance(pixels[1], list) is True:
        pixels_left, pixels_right = sorted(pixels)
    elif isinstance(pixels[0], int) and isinstance(pixels[1], list) is True:
        pixels_left, pixels_right = sorted([[pixels[0]], pixels[1]])
    elif isinstance(pixels[0], list) and isinstance(pixels[1], int) is True:
        pixels_left, pixels_right = sorted([pixels[0], [pixels[1]]])
    elif isinstance(pixels[0], int) and isinstance(pixels[1], int) is True:
        pixels_left = pixels
        pixels_right = pixels

    for q in pixels_left:
        for w in pixels_right:
            if w <= q:
                continue
            deltas_all["{},{}".format(q, w)] = []
            # Find ends of cycles
            cycle_ends = np.argwhere(data[0].T[0] == -2)
            cycle_ends = np.insert(cycle_ends, 0, 0)
            # First pixel in the pair
            tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
            pix1 = np.where(data[tdc1].T[0] == pix_c1)[0]
            # Second pixel in the pair
            tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
            pix2 = np.where(data[tdc2].T[0] == pix_c2)[0]
            # Get timestamp for both pixels in the given cycle

            for cyc in range(len(cycle_ends) - 1):
                pix1_ = pix1[
                    np.logical_and(
                        pix1 >= cycle_ends[cyc], pix1 < cycle_ends[cyc + 1]
                    )
                ]
                if not np.any(pix1_):
                    continue
                pix2_ = pix2[
                    np.logical_and(
                        pix2 >= cycle_ends[cyc], pix2 < cycle_ends[cyc + 1]
                    )
                ]
                if not np.any(pix2_):
                    continue
                # Calculate delta t
                tmsp1 = data[tdc1].T[1][
                    pix1_[np.where(data[tdc1].T[1][pix1_] > 0)[0]]
                ]
                tmsp2 = data[tdc2].T[1][
                    pix2_[np.where(data[tdc2].T[1][pix2_] > 0)[0]]
                ]
                for t1 in tmsp1:
                    deltas = tmsp2 - t1
                    ind = np.where(np.abs(deltas) < delta_window)[0]
                    deltas_all["{},{}".format(q, w)].extend(deltas[ind])

    return deltas_all
