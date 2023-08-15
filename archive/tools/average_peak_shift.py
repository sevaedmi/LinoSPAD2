"""Script for plotting a grid 4x4 of delta t for different pairs of pixels.
Saves peak positions to a 'csv' file.

"""

import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack as f_up

plt.ioff()

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "Ne lamp ext trig/setup 2/3 ms acq window"
)

os.chdir(path)

DATA_FILES = glob.glob("*.dat*")

for filename in DATA_FILES:
    data = f_up.unpack_binary_512(filename)

    data_251 = data[30]  # 251st pixel
    data_252 = data[31]  # 253st pixel
    data_253 = data[32]  # 254st pixel
    data_254 = data[33]  # 254st pixel
    data_255 = data[34]  # 254st pixel

    pixel_numbers = np.arange(30, 35, 1)

    all_data = np.vstack((data_251, data_252, data_253, data_254, data_255))

    plt.rcParams.update({"font.size": 20})
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))

    # save pixel pairs and their peaks
    peaks_num = []
    peaks_pos = []

    for q in range(5):
        for w in range(5):
            if w <= q:
                continue

            data_pair = np.vstack((all_data[q], all_data[w]))

            minuend = len(data_pair) - 1  # i=255
            lines_of_data = len(data_pair[0])
            subtrahend = len(data_pair)  # k=254
            timestamps = 512  # lines of data in the acq cycle

            output = []

            for i in tqdm(range(minuend)):
                acq = 0  # number of acq cycle
                for j in range(lines_of_data):
                    if data_pair[i][j] == -1:
                        continue
                    if j % 512 == 0:
                        acq = acq + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 153-45 etc.
                        for p in range(timestamps):
                            n = 512 * (acq - 1) + p
                            if data_pair[k][n] == -1:
                                continue
                            elif data_pair[i][j] - data_pair[k][n] > 1e5:  #
                                continue
                            elif data_pair[i][j] - data_pair[k][n] < -1e5:
                                continue
                            else:
                                output.append(data_pair[i][j] - data_pair[k][n])

            bins = np.arange(np.min(output), np.max(output), 17.857 * 100)
            axs[q][w - 1].set_xlabel("delta_t [ps]")
            axs[q][w - 1].set_ylabel("Timestamps [-]")

            n, b, p = axs[q][w - 1].hist(output, bins=bins)
            # find position of the histogram peak
            n_max = np.argmax(n)
            arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")

            axs[q][w - 1].set_title(
                "Pixels {p1}-{p2}\nPeak position {pp}".format(
                    p1=pixel_numbers[q], p2=pixel_numbers[w], pp=arg_max
                )
            )

            peaks_num.append(
                "{p1}-{p2}".format(p1=pixel_numbers[q], p2=pixel_numbers[w])
            )
            peaks_pos.append(arg_max)

    peaks = np.vstack((peaks_num, peaks_pos))
    # open csv file with results and add a column
    print("\nSaving the data to the 'results' folder.")
    peaks_save = pd.DataFrame(peaks)

    try:
        os.chdir("results")
    except Exception:
        os.mkdir("results")
        os.chdir("results")
    try:
        filename_out = glob.glob("*Peaks_positions*")[0]
    except Exception:
        with open("Peaks_positions.csv", "w"):
            filename_out = glob.glob("*Peaks_positions*")[0]
            pass

    peaks_save.to_csv(filename_out, header=None, mode="a", index=False)

    try:
        os.chdir("delta_t")
    except Exception:
        os.mkdir("delta_t")
        os.chdir("delta_t")
    fig.tight_layout()  # for perfect spacing between the plots
    plt.savefig("{name}_delta_t_grid.png_noise".format(name=filename))
    plt.pause(0.1)
    plt.close("all")
    os.chdir("../..")
