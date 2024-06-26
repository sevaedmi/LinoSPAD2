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

from LinoSPAD2.functions import utils


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
        Array for transforming the pixel address in terms of TDC (0 to 3)
        to pixel number in terms of half of the sensor (0 to 255).
    delta_window : float, optional
        Width of the time window for counting timestamp differences.
        The default is 50e3 (50 ns).

    Returns
    -------
    deltas_all : dict
        Dictionary containing timestamp differences for each pair of pixels.

    """
    deltas_all = {}

    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    # Find ends of cycles
    cycle_ends = np.argwhere(data[0].T[0] == -2)
    cycle_ends = np.insert(cycle_ends, 0, 0)

    for q in pixels_left:
        # First pixel in the pair
        tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
        pix1 = np.where(data[tdc1].T[0] == pix_c1)[0]
        for w in pixels_right:
            if w <= q:
                continue
            deltas_all["{},{}".format(q, w)] = []

            # Second pixel in the pair
            tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
            pix2 = np.where(data[tdc2].T[0] == pix_c2)[0]

            # Go over cycles, getting data for the appropriate cycle
            # only
            for cyc in range(len(cycle_ends) - 1):
                slice_from = cycle_ends[cyc]
                slice_to = cycle_ends[cyc + 1]
                pix1_slice = pix1[(pix1 >= slice_from) & (pix1 < slice_to)]
                if not np.any(pix1_slice):
                    continue
                pix2_slice = pix2[(pix2 >= slice_from) & (pix2 < slice_to)]
                if not np.any(pix2_slice):
                    continue

                # Calculate delta t
                tmsp1 = data[tdc1].T[1][pix1_slice]
                tmsp1 = tmsp1[tmsp1 > 0]
                tmsp2 = data[tdc2].T[1][pix2_slice]
                tmsp2 = tmsp2[tmsp2 > 0]
                for t1 in tmsp1:
                    deltas = tmsp2 - t1
                    ind = np.where(np.abs(deltas) < delta_window)[0]
                    deltas_all["{},{}".format(q, w)].extend(deltas[ind])

    return deltas_all
