from struct import unpack

import pandas as pd


def unpack_binary_df(
    filename, timestamps: int = 512, apply_mask: bool = True, cut_empty: bool = True
):
    """Unpacks the 'dat' files with a given number of timestamps per acquisition cycle
    per pixel into a pandas dataframe. The fastest unpacking compared to others.

    Parameters
    ----------
    filename : str
        The 'dat' binary-encoded datafile.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.
    apply_mask : bool, optional
        Switch for masking the warm/hot pixels. Default is True.
    cut_empty : bool, optional
        Switch for appending '-1', or either non-valid or empty timestamps, to the
        output. Default is True.

    Returns
    -------
    timestamps : pandas.DataFrame
        A dataframe with two columns: first is the pixel number, second are the
        timestamps.

    """

    mask = [
        15,
        16,
        29,
        39,
        40,
        50,
        52,
        66,
        73,
        93,
        95,
        96,
        98,
        101,
        109,
        122,
        127,
        196,
        210,
        231,
        236,
        238,
    ]

    timestamp_list = list()
    pixels_list = list()
    cycles_list = list()
    acq = 1
    i = 0
    cycles = timestamps * 256

    with open(filename, "rb") as f:
        while True:
            i += 1
            rawpacket = f.read(4)
            if not rawpacket:
                break
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:
                # timestamps cover last 28 bits of the total 32
                timestamp = packet[0] & 0xFFFFFFF
                # 17.857 ps - average bin size
                timestamp = timestamp * 17.857
                pixels_list.append(int(i / 512) + 1)
                timestamp_list.append(timestamp)
                cycles_list.append(acq)
            elif cut_empty is False:
                timestamp_list.append(-1)
                pixels_list.append(int(i / 512 + 1))
                cycles_list.append(acq)
            if i == cycles:
                i = 0
                acq += 1
    dic = {"Pixel": pixels_list, "Timestamp": timestamp_list, "Cycle": cycles_list}
    timestamps = pd.DataFrame(dic)
    timestamps = timestamps[~timestamps["Pixel"].isin(mask)]

    return timestamps


# TODO: add raise error for checking the 1) tidyness of the input dataframes 2) for
# absence of "-1" timestamps
def calc_diff_df(
    data_1,
    data_2,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
):
    """
    Function for calculating timestamp differences for a given pair of pixels. Input
    dataframes for both pixels should be 1) tidy and 2) without the invalid "-1"
    timestamps.

    Parameters
    ----------
    data_1 : pandas.DataFrame
        Tidy dataframe with data from the 1st pixel of the pair.
    data_2 : pandas.DataFrame
        Tidy dataframe with data from the 2nd pixel of the pair.
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
    output : pandas.DataFrame
        A dataframe of timestamp differences for the given pair of pixels.

    """

    output = []
    cycles = data_1.Cycle.max()
    c = 1

    while c != cycles:
        # with .values turn dataframes to numpy arrays -> faster computation time
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
