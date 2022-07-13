"""Plots number of valid timestamps in each pixel for each 'dat' file in the
folder.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

The output is saved in the `results` directory, in the case there is no such
folder, it is created where the data are stored.

This file can also be imported as a module and contains the following
functions:

    * plot_valid_per_pixel - plot number of valid timestamps in each pixel

"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

# =============================================================================
# Data collected with the sensor cover but without the optical fiber attached
# =============================================================================


def plot_valid_per_pixel(path, lod, scale: str = 'linear'):
    '''Plots number of valid timestamps in each pixel for each 'dat' file in
    given folder. The plots are saved as 'png' in the 'results' folder. In
    the case there is no such folder, it is created where the data files are.

    Parameters
    ----------
    path : str
        Location of the 'dat' files with the data from LinoSPAD2.
    lod : int
        Lines of data per acquistion cycle.
    scale : str
        Use 'log' for logarithmic scale, leave empty for linear.

    Returns
    -------
    None.

    '''
    os.chdir(path)

    DATA_FILES = glob.glob('*acq*'+'*dat*')

    valid_per_pixel = np.zeros(256)

    for i, num in enumerate(DATA_FILES):
        data_matrix = f_up.unpack_binary_flex(num, lod)
        for j in range(len(data_matrix)):
            valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

        plt.ioff()
        plt.figure(figsize=(16, 10))
        plt.rcParams.update({"font.size": 20})
        plt.title("{}".format(num))
        plt.xlabel("Pixel [-]")
        plt.ylabel("Valid timestamps [-]")
        if scale == 'log':
            plt.yscale('log')
        plt.plot(valid_per_pixel, 'o')

        try:
            os.chdir("results")
        except Exception:
            os.mkdir("results")
            os.chdir("results")

        plt.savefig("{}.png".format(num))
        os.chdir("..")
