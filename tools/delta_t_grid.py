"""Script for plotting a grid 4x4 of delta t for different pairs of pixels.

"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/Ne lamp ext trig/setup 2/FW 2208/3.99 ms"

os.chdir(path)

# ==== Give the range of pixels to analyze! =====
pix = np.array((155, 156, 157, 158, 159))

filename = glob.glob('*.dat*')[0]

data = f_up.unpack_binary_512(filename)

data_1 = data[pix[0]]  # 1st pixel
data_2 = data[pix[1]]  # 2nd pixel
data_3 = data[pix[2]]  # 3d pixel
data_4 = data[pix[3]]  # 4th pixel
data_5 = data[pix[4]]  # 5th pixel

pixel_numbers = np.arange(pix[0], pix[-1]+1, 1)

all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(4, 4, figsize=(24, 24))

for q in range(5):
    for w in range(5):
        if w <= q:
            continue

        data_pair = np.vstack((all_data[q], all_data[w]))

        minuend = len(data_pair)-1  # i=255
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
                        n = 512*(acq-1) + p
                        if data_pair[k][n] == -1:
                            continue
                        elif data_pair[i][j] - data_pair[k][n] > 1e5:  #
                            continue
                        elif data_pair[i][j] - data_pair[k][n] < -1e5:
                            continue
                        else:
                            output.append(data_pair[i][j]
                                          - data_pair[k][n])

        bins = np.arange(np.min(output), np.max(output), 17.857*100)
        axs[q][w-1].set_xlabel('\u0394t [ps]')
        axs[q][w-1].set_ylabel('Timestamps [-]')

        n, b, p = axs[q][w-1].hist(output, bins=bins)
        # find position of the histogram peak
        n_max = np.argmax(n)
        arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")

        axs[q][w-1].set_title('Pixels {p1}-{p2}\nPeak position {pp}'
                              .format(p1=pixel_numbers[q],
                                      p2=pixel_numbers[w],
                                      pp=arg_max))

os.chdir("results")
fig.tight_layout()  # for perfect spacing between the plots
plt.savefig("{name}_delta_t_grid.png".format(name=filename))
