"""Script for fitting peaks in warm pixels delta ts with a gaussian function.

"""

import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from functions import unpack as f_up


def gauss(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "Ar lamp/FW 2208"
)

os.chdir(path)

filename = glob.glob("*.dat*")[0]

lines_of_data = 512

pix = 236

data = f_up.unpack_binary_flex(filename, 512)

data_cut = np.vstack((data[pix], data[pix + 1]))

minuend = len(data_cut)
timestamps_total = len(data_cut[0])
subtrahend = len(data_cut)
timestamps = lines_of_data

output = []

for i in tqdm(range(minuend)):
    acq = 0  # number of acq cycle
    for j in range(timestamps_total):
        if j % lines_of_data == 0:
            acq = acq + 1  # next acq cycle
        if data_cut[i][j] == -1:
            continue
        for k in range(subtrahend):
            if k <= i:
                continue  # to avoid repetition: 2-1, 53-45
            for p in range(timestamps):
                n = lines_of_data * (acq - 1) + p
                if data_cut[k][n] == -1:
                    continue
                elif data_cut[i][j] - data_cut[k][n] > 2.5e3:
                    continue
                elif data_cut[i][j] - data_cut[k][n] < -2.5e3:
                    continue
                else:
                    output.append(data_cut[i][j] - data_cut[k][n])

if "Ne" and "540" in path:
    chosen_color = "seagreen"
elif "Ne" and "656" in path:
    chosen_color = "orangered"
elif "Ar" in path:
    chosen_color = "mediumslateblue"
else:
    chosen_color = "salmon"

try:
    bins = np.arange(np.min(output), np.max(output), 17.857)
except Exception:
    print(1)

plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel("\u0394t [ps]")
plt.ylabel("Timestamps [-]")
n, b, p = plt.hist(output, bins=bins, color=chosen_color)

try:
    n_max = np.argmax(n)
    arg_max = (bins[n_max] + bins[n_max + 1]) / 2
except Exception:
    arg_max = None
    pass

sigma = 200

par, covariance = curve_fit(gauss, b[:-1], n, p0=[max(n), arg_max, sigma])
fit_plot = gauss(b, par[0], par[1], par[2])

b_fit = b
n_fit = n

plt.figure(figsize=(16, 10))
plt.xlabel("\u0394t [ps]")
plt.ylabel("Timestamps [-]")
plt.hist(output, bins=bins, color=chosen_color, histtype="step", label="data")
plt.plot(
    b_fit,
    fit_plot,
    "-",
    color="cadetblue",
    label="fit\n\u03BC={p1} ps\n"
    "\u03C3={p2} ps".format(p1=format(par[1], ".2f"), p2=format(par[-1], ".2f")),
)
plt.legend(loc="best")
