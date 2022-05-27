"""Plot cross-talk rate distribution in the LinoSPAD2 sensor.

Uses the data from a '.csv' file, plots average cross-talk rate vs pixel.

This file can also be imported as a module and contains the following
functions:

    * ctr_dist - calculates the cross-talk rate distribution in the sensor
"""

import os
import glob
from matplotlib import pyplot as plt
import numpy as np


def ctr_dist(path):
    """Plots cross-talk rate distribution in the sensor.
    The plot is saved as a '.pdf' file.

    Parameters
    ----------
    path : str
        Location of the data file.

    Returns
    -------
    None.

    """

    os.chdir(path)
    path_res = glob.glob('*results*')[0]
    os.chdir(path_res)
    data_csv = glob.glob('*Cross-talk by pixel*')[0]
    data = np.genfromtxt(data_csv, delimiter=',', skip_header=1)

    pixel = np.array([row[0] for row in data])
    ct_rate = np.array([row[1:] for row in data])

    ct_rate_average = np.zeros(len(ct_rate))

    for i in range(len(ct_rate)):
        ct_rate_average[i] = np.sum(ct_rate[i]) / len(ct_rate[0])

    plt.figure(figsize=(16, 10))
    plt.rcParams.update({'font.size': 22})
    plt.xlabel("Pixel [-]")
    plt.ylabel("Cross-talk rate [%]")
    plt.plot(pixel, ct_rate_average, 'o', color='salmon')
    plt.savefig("Cross-talk_distribution.pdf")
