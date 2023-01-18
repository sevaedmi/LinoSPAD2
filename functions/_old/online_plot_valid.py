"""Script for online printing of histograms of valid timestamps vs pixel
number. Waits for a new data file, processes it when the file appears, shows
and saves the output figure. Then the cycle repeats.

"""

import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from functions import unpack as f_up


def online_plot_valid(path, pix_range, timestamps: int = 512, frame_rate: float = 0.1):
    """
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
    frame_rate : float, optional
        Time for the script to wait for a new file.
    Returns
    -------
    None.

    """

    os.chdir(path)

    last_file_ctime = 0

    pixels = np.arange(0, 256, 1)

    plt.ion()

    print(
        "==============================\n"
        "Online plotting of timestamps\n"
        "=============================="
    )

    fig = plt.figure(figsize=(11, 7))
    plt.rcParams.update({"font.size": 22})

    while True:

        DATA_FILES = glob.glob("*.dat*")
        try:
            last_file = max(DATA_FILES, key=os.path.getctime)
        except ValueError:
            print("Waiting for a file")
            plt.pause(frame_rate)
            continue
        new_file_ctime = os.path.getctime(last_file)

        if new_file_ctime > last_file_ctime:

            waiting = False

            last_file_ctime = new_file_ctime

            print("Analysing the last file.")
            data = f_up.unpack_numpy(last_file, timestamps)

            valid_per_pixel = np.zeros(256)

            for j in range(len(data)):
                valid_per_pixel[j] = len(np.where(data[j] > 0)[0])
            peak = np.max(valid_per_pixel[pix_range])

            if "Ne" and "540" in path:
                chosen_color = "seagreen"
            elif "Ne" and "656" in path:
                chosen_color = "orangered"
            elif "Ar" in path:
                chosen_color = "mediumslateblue"
            else:
                chosen_color = "salmon"
            try:
                plot.set_xdata(pixels)
                plot.set_ydata(valid_per_pixel)
                plt.title("Peak is {}".format(peak))  # show new peak value

                # re-drawing the figure, recalculating the axes limits
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()

                # to flush the GUI events
                fig.canvas.flush_events()
                plt.pause(frame_rate)
            except NameError:
                ax = fig.add_subplot(111)
                (plot,) = ax.plot(pixels, valid_per_pixel, "o", color=chosen_color)
                plt.rcParams.update({"font.size": 22})
                plt.title("Peak is {}".format(peak))
                plt.xlabel("Pixel [-]")
                plt.ylabel("Valid timestamps [-]")
                plt.yscale("log")
                plt.show()
                plt.pause(frame_rate)
        else:
            if waiting is False:
                print("Waiting for new file.")
            waiting = True
            last_file_ctime = new_file_ctime
            plt.pause(frame_rate)
            continue
