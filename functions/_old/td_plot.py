"""Plot a histogram of timestamp differences from LinoSPAD2.

The output is saved in the `results` directory.

This file can also be imported as a module and contains the following
functions:

    * plot_diff - plots a histogram of the LinoSPAD2 timestamp differences

"""

import glob
import os
import sys

import numpy as np
from matplotlib import pyplot as plt


def plot_diff(path, show_fig: bool = False):
    """Plots a histogram of LinoSPAD2 timestamps differences.

    Parameters
    ----------
    path : str
        Location of the 'csv' file with timestamp differences from LinoSPAD2.
        By default, the file is in the 'results' folder where data are
        located.
    show_fig : bool
        Should the script show the figure or not. Default value is 'False'.

    Returns
    -------
    None.

    """

    os.chdir(path)

    filename = glob.glob("*timestamp_diff*")

    if filename == []:
        print("No 'csv' file with timestamp differences was found.")
        sys.exit()

    Data = np.genfromtxt(filename[0], delimiter=",", skip_header=1)

    # dimenstions for transformed data array
    dim_x = len(Data)
    dim_y = len(Data[0])

    # put data into a 1D array
    data = np.reshape(Data, dim_x * dim_y)

    # cut the nan values that may appear due to the different column size
    # in the timestamp-differences csv file
    data = np.delete(data, np.where(np.isnan(data) == True)[0])

    mi = np.min(data)
    ma = np.max(data)

    bins = np.arange(mi, ma, 100)  # bin size is 100 ps

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 20})
    plt.xlabel("Time [ps]")
    plt.ylabel("Hits [-]")
    plt.hist(data, bins=bins)
    plt.savefig("Time-diff hist.pdf")
    print("The output is saved in the '{}' folder.".format(path))
