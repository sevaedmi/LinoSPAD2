"""Script for plotting a 4x4 grid of delta t for data sets from the setups
w/o optics and from w/ a polarizer.

"""

import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ar lamp/FW 2208"

os.chdir(path)

# ==== Give the range of pixels to analyze! =====
pix = np.array((156, 157, 158, 159, 160))

filename = glob.glob('*.dat*')[0]
filename2 = glob.glob('*.dat*')[1]

data = f_up.unpack_binary_512(filename)
data2 = f_up.unpack_binary_512(filename2)

data_1 = data[pix[0]]  # 1st pixel
data_2 = data[pix[1]]  # 2nd pixel
data_3 = data[pix[2]]  # 3d pixel
data_4 = data[pix[3]]  # 4th pixel
data_5 = data[pix[4]]  # 5th pixel

data2_1 = data2[pix[0]]  # 1st pixel
data2_2 = data2[pix[1]]  # 2nd pixel
data2_3 = data2[pix[2]]  # 3d pixel
data2_4 = data2[pix[3]]  # 4th pixel
data2_5 = data2[pix[4]]  # 5th pixel

pixel_numbers = np.arange(pix[0], pix[-1]+1, 1)

all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))
all_data2 = np.vstack((data2_1, data2_2, data2_3, data2_4, data2_5))

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(4, 4, figsize=(24, 24))

y_max_all = 0

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
                        elif data_pair[i][j] - data_pair[k][n] > 2.5e3:  #
                            continue
                        elif data_pair[i][j] - data_pair[k][n] < -2.5e3:
                            continue
                        else:
                            output.append(data_pair[i][j]
                                          - data_pair[k][n])
        try:
            bins = np.arange(np.min(output), np.max(output), 17.857*2)
        except Exception:
            continue

        # plt.gcf()
        axs[q][w-1].set_xlabel('\u0394t [ps]')
        axs[q][w-1].set_ylabel('Timestamps [-]')
        n, b, p = axs[q][w-1].hist(output, bins=bins, histtype='step',
                                   color="slateblue")
        # find position of the histogram peak
        try:
            n_max = np.argmax(n)
        except Exception:
            continue
        arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")

        y_max = np.max(n)
        if y_max_all < y_max:
            y_max_all = y_max
        
        axs[q][w-1].set_ylim(0, y_max+10)
        # axs[q][w-1].set_xlim(-2.5e3, 2.5e3)

        axs[q][w-1].set_title('Pixels {p1}-{p2}\nPeak position {pp}'
                              .format(p1=pixel_numbers[q],
                                      p2=pixel_numbers[w],
                                      pp=arg_max))


for q in range(5):
    for w in range(5):
        if w <= q:
            continue

        data_pair2 = np.vstack((all_data2[q], all_data2[w]))

        minuend = len(data_pair2)-1  # i=255
        lines_of_data = len(data_pair2[0])
        subtrahend = len(data_pair2)  # k=254
        timestamps = 512  # lines of data in the acq cycle

        output2 = []

        for i in tqdm(range(minuend)):
            acq = 0  # number of acq cycle
            for j in range(lines_of_data):
                if data_pair2[i][j] == -1:
                    continue
                if j % 512 == 0:
                    acq = acq + 1  # next acq cycle
                for k in range(subtrahend):
                    if k <= i:
                        continue  # to avoid repetition: 2-1, 153-45 etc.
                    for p in range(timestamps):
                        n = 512*(acq-1) + p
                        if data_pair2[k][n] == -1:
                            continue
                        elif data_pair2[i][j] - data_pair2[k][n] > 2.5e3:  #
                            continue
                        elif data_pair2[i][j] - data_pair2[k][n] < -2.5e3:
                            continue
                        else:
                            output2.append(data_pair2[i][j]
                                          - data_pair2[k][n])
        try:
            bins = np.arange(np.min(output2), np.max(output2), 17.857*2)
        except Exception:
            continue
        axs[q][w-1].set_xlabel('\u0394t [ps]')
        axs[q][w-1].set_ylabel('Timestamps [-]')

        # plt.gcf()
        try:
            n, b, p = axs[q][w-1].hist(output2, bins=bins, histtype='step',
                                   color='orange')
        except Exception:
            continue
        # find position of the histogram peak
        try:
            n_max = np.argmax(n)
        except Exception:
            continue
        arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")

        y_max = np.max(n)
        if y_max_all < y_max:
            y_max_all = y_max
        
        axs[q][w-1].set_ylim(0, y_max+10)
        # axs[q][w-1].set_xlim(-2.5e3, 2.5e3)

        axs[q][w-1].set_title('Pixels {p1}-{p2}\nPeak position {pp}'
                              .format(p1=pixel_numbers[q],
                                      p2=pixel_numbers[w],
                                      pp=arg_max))



for q in range(5):
    for w in range(5):
        if w <= q:
            continue
        axs[q][w-1].set_ylim(0, y_max_all+10)
plt.show()

os.chdir("results/delta_t")
# fig = plt.gcf()
plt.tight_layout()  # for perfect spacing between the plots
plt.savefig("{name}_delta_t_grid_2_data_sets.png".format(name=filename))
