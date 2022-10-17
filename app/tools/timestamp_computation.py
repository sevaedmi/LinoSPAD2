import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import app.tools.unpack_data as unpk

def get_nmr_validtimestamps(path, pix_range, timestamps: int = 512):
    '''
    Real-time plotting of number of valid timestamps from the last data
    file. Waits for a new file then analyzes it and shows the plot. The
    output is saved in the "results/online" directory. In the case the folder
    does not exist, it is created.

    Parameters
    ----------
    path : str
        Path to where the data files are.
    pix_range : array-like
        Range of pixels for which the maximum is calculated and is then shown
        in the plot.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    Returns
    -------
    None.

    '''

    data = unpk.unpack_binary_flex(path, 512)

    valid_per_pixel = np.zeros(256)

    for j in range(len(data)):
        valid_per_pixel[j] = len(np.where(data[j] > 0)[0])

    peak = np.max(valid_per_pixel[pix_range])

    return valid_per_pixel, peak



def plot_pixel_hist(path, pix1, timestamps: int = 512, show_fig: bool = False):
    '''
    Plots a histogram for each pixel in a preset range.

    Parameters
    ----------
    path : str
        Path to data file.
    pix1 : array-like
        Array of pixels indices. Preferably pixels where the peak is.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.

    Returns
    -------
    None.

    '''

    os.chdir(path)

    DATA_FILES = glob.glob('*.dat*')

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()

    for i, num in enumerate(DATA_FILES):

        print("=====================================================\n"
              "Plotting pixel histograms, Working on {}\n"
              "====================================================="
              .format(num))

        data = f_up.unpack_binary_flex(num, timestamps)

        if pix1 is None:
            pixels = np.arange(145, 165, 1)
        else:
            pixels = pix1

        for i, pixel in enumerate(pixels):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({'font.size': 22})
            bins = np.arange(0, 4e9, 17.867*1e6)  # bin size of 17.867 us
            plt.hist(data[pixel], bins=bins, color="teal")
            plt.xlabel("Time [ms]")
            plt.ylabel("Counts [-]")
            plt.title("Pixel {}".format(pixel))
            try:
                os.chdir("results/single pixel histograms")
            except Exception:
                os.mkdir("results/single pixel histograms")
                os.chdir("results/single pixel histograms")
            plt.savefig("{file}, pixel {pixel}.png".format(file=num,
                                                           pixel=pixel))
            os.chdir("../..")
