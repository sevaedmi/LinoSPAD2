import os
import numpy as np
import glob
import time
from matplotlib import pyplot as plt
from functions import unpack as f_up

timestamps = 512

pix_range = np.arange(140, 161, 1)

pixels = np.arange(0, 256, 1)

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ne lamp/FW 2208/653 nm"

path_save = path + "/results/online"

os.chdir(path)

last_file_ctime = 0
plt.ion()

print("===========================\n"
      "Online plotting of timestamps"
      "===========================\n")

plt.ion()

fig = plt.figure()

while True:

    DATA_FILES = glob.glob('*.dat*')
    try:
        last_file = max(DATA_FILES, key=os.path.getctime)
    except Exception:
        print("Waiting for a file")
        time.sleep(5)
        continue

    new_file_ctime = os.path.getctime(last_file)

    if new_file_ctime <= last_file_ctime:
        print("Waiting for new file.")
        last_file_ctime = new_file_ctime
        time.sleep(5)
        continue

# =========================================================================
# Data analysis
# =========================================================================
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
     
        # re-drawing the figure
        fig.canvas.draw()
         
        # to flush the GUI events
        fig.canvas.flush_events()
        time.sleep(5)
    except NameError:
        ax = fig.add_subplot(111)
        plot, = ax.plot(pixels, valid_per_pixel)
        plt.rcParams.update({"font.size": 20})
        plt.title("Peak is {}".format(peak))
        plt.xlabel("Pixel [-]")
        plt.ylabel("Valid timestamps [-]")
        plt.yscale('log')
        plt.show()
        plt.pause(1)