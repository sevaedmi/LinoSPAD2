import glob
import os

import numpy as np
from matplotlib import pyplot as plt

from functions import unpack as f_up

timestamps = 512

frame_rate = 0.1  # how often should the script look for the new file

pix_range = np.arange(120, 141, 1)

# cut the noisy pixel from the peak
pix_range = np.delete(pix_range, (np.where(pix_range == 122)[0]))

pixels = np.arange(0, 256, 1)

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "Ne lamp/FW 2208/653 nm"
)

path_save = path + "/results/online"

os.chdir(path)

last_file_ctime = 0

plt.ion()

print(
    "===========================\n"
    "Online plotting of timestamps\n"
    "==========================="
)

fig = plt.figure(figsize=(11, 7))
plt.rcParams.update({"font.size": 22})

while True:
    DATA_FILES = glob.glob("*.dat*")
    try:
        last_file = max(DATA_FILES, key=os.path.getctime)
    except Exception:
        print("Waiting for a file")
        plt.pause(frame_rate)
        continue

    new_file_ctime = os.path.getctime(last_file)

    if new_file_ctime > last_file_ctime:
        last_file_ctime = new_file_ctime

        print("Analysing the last file.")
        data = f_up.unpack_binary_flex(last_file, timestamps)

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
        print("Waiting for new file.")
        last_file_ctime = new_file_ctime
        plt.pause(frame_rate)
        continue
