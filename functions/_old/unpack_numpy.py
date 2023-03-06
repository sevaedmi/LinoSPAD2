import numpy as np

# def unpack_numpy(filename, timestamps: int = 512):
#     """
#     Function for unpacking binary data based on the numpy library.

#     Parameters
#     ----------
#     filename : str
#         Name of the file with the binary-encoded data.
#     timestamps : int, optional
#         Number of timestamps per acquisition cycle per pixel. Default is 512.

#     Returns
#     -------
#     data_matrix : array_like
#         A 2D matrix (256 pixels by timestamps X number-of-cycles) of
#         timestamps.

#     """
#     rawFile = np.fromfile(filename, dtype=np.uint32)  # read data
#     data = (rawFile & 0xFFFFFFF).astype(int) * 17.857  # Multiply with the lowes bin
#     data[np.where(rawFile < 0x80000000)] = -1  # Mask not valid data
#     cycles = int(len(data) / timestamps / 256)  # number of cycles,
#     data_matrix = (
#         data.reshape((timestamps, cycles * 256), order="F")
#         .reshape((timestamps, 256, -1), order="F")
#         .transpose((0, 2, 1))
#         .reshape((-1, 256), order="F")
#         .transpose()
#     )  # reshape the matrix
#     return data_matrix


def unpack_numpy(filename, timestamps: int = 512):
    """
    Function for unpacking the binary-encoded data from the LinoSPAD2
    based on the numpy library. Currently, the fastest option for
    unpacking.

    Parameters
    ----------
    filename : str
        Name of the file with the data.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
        The default is 512.

    Returns
    -------
    output : ndarray
        A 2D matrix (256 x timestamps*number_of_cycles) of timestamps.

    Examples
    --------
    To see how the function works, consider a setup of 4 pixels,
    5 timestamps per pixel per acquisition cycle, 3 cycles total.
    The output of the LinoSPAD2 data acquisition software is pixel
    by pixel, cycle after cycle. Therefore, for timestamps from 0
    to 59, timestamps 0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 40, 41, 42,
    43, 44 are from the first pixel.

    >>> a = np.arange(4*5*3)
    >>> a
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    >>> b = a.reshape(3, 4, 5)
    >>> b
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
           [[20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39]],
           [[40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59]]])
    >>> c = b.transpose((1, 0, 2))
    >>> c
    array([[[ 0,  1,  2,  3,  4],
            [20, 21, 22, 23, 24],
            [40, 41, 42, 43, 44]],
           [[ 5,  6,  7,  8,  9],
            [25, 26, 27, 28, 29],
            [45, 46, 47, 48, 49]],
           [[10, 11, 12, 13, 14],
            [30, 31, 32, 33, 34],
            [50, 51, 52, 53, 54]],
           [[15, 16, 17, 18, 19],
            [35, 36, 37, 38, 39],
            [55, 56, 57, 58, 59]]])
    >>> d = c.reshape(4, 3*5)
    >>> d
    array([[ 0,  1,  2,  3,  4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44],
           [ 5,  6,  7,  8,  9, 25, 26, 27, 28, 29, 45, 46, 47, 48, 49],
           [10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50, 51, 52, 53, 54],
           [15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59]])

    """
    # read data by 32 bit words
    rawFile = np.fromfile(filename, dtype=np.uint32)
    # lowest 28 bits are the timestamp; convert to ps
    data = (rawFile & 0xFFFFFFF).astype(np.longlong) * 17.857
    # mask nonvalid data with '-1'
    data[np.where(rawFile < 0x80000000)] = -1
    # number of acquisition cycles
    cycles = int(len(data) / timestamps / 256)

    data_matrix = (
        data.reshape(cycles, 256, timestamps)
        .transpose((1, 0, 2))
        .reshape(256, timestamps * cycles)
    )

    return data_matrix
