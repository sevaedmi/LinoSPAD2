""" Function for calculating timestamp differences for the given pair of
pixels.

This file can also be imported as a module and contains the following
functions:

    * calculate differences - calculate timestamp differences for a given
    pair of pixels. Returns an array of differences.

"""
import pandas as pd


def calc_diff(
    data_pair,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """
    Function for calculating timestamp differences for a given pair of pixels.

    Parameters
    ----------
    data_pair : array-like
        Data from 2 pixels.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition window. The default is
        512.
    range_left : float, optional
        Left limit for the timestamp differences. Values below that are not
        taken into account and the differences are not returned. The default
        is -2.5e3.
    range_right : float, optional
        Right limit for the timestamp differences. Values above that are not
        taken into account and the differences are not returned. The default
        is 2.5e3.

    Returns
    -------
    output : array-like
        An array of timestamp differences for the given pair of pixels.

    """

    minuend = len(data_pair)
    timestamps_total = len(data_pair[0])
    subtrahend = len(data_pair)

    output = []

    for i in range(minuend):
        acq = 0  # number of acq cycle
        for j in range(timestamps_total):
            if j % timestamps == 0:
                acq = acq + 1  # next acq cycle
            if data_pair[i][j] == -1:
                continue
            for k in range(subtrahend):
                if k <= i:
                    continue  # to avoid repetition: 2-1, 53-45
                for p in range(timestamps):
                    n = timestamps * (acq - 1) + p
                    if data_pair[k][n] == -1:
                        continue
                    # elif data_pair[i][j] - data_pair[k][n] > range_right:
                    #     continue
                    # elif data_pair[i][j] - data_pair[k][n] < range_left:
                    #     continue
                    # else:
                    #     output.append(data_pair[i][j] - data_pair[k][n])
                    else:
                        delta = data_pair[i][j] - data_pair[k][n]
                        if delta > range_right:
                            continue
                        elif delta < range_left:
                            continue
                        else:
                            output.append(delta)
    return output


def calc_diff_df(
    data_1,
    data_2,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """
    Function for calculating timestamp differences for a given pair of pixels.

    Parameters
    ----------
    data_1 : pandas.Series
        Data from the 1st pixel of the pair.
    data_2 : pandas.Series
        Data from the 2nd pixel of the pair.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition window. The default is
        512.
    range_left : float, optional
        Left limit for the timestamp differences. Values below that are not
        taken into account and the differences are not returned. The default
        is -2.5e3.
    range_right : float, optional
        Right limit for the timestamp differences. Values above that are not
        taken into account and the differences are not returned. The default
        is 2.5e3.

    Returns
    -------
    output : array-like
        An array of timestamp differences for the given pair of pixels.

    """

    output = []
    cycles = data_1.Cycle.max()
    c = 1

    while c != cycles:
        data_1_c = data_1.Timestamp[data_1.Cycle == c].values
        data_2_c = data_2.Timestamp[data_2.Cycle == c].values
        for i in range(len(data_1_c)):
            for j in range(len(data_2_c)):
                delta = data_1_c[i] - data_2_c[j]
                if delta < range_left:
                    continue
                elif delta > range_right:
                    continue
                else:
                    output.append(delta)
        c += 1
    dic = {"Delta t": output}
    output = pd.DataFrame(dic)

    return output
