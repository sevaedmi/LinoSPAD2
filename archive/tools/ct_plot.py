import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem

path = "C:/Users/bruce/Documents/Quantum astrometry/CT"

os.chdir(path)

# file = glob.glob("*CT_data.csv*")[0]
file_name = "CT_data_Ne_585_134,140.csv"
# file_name = "CT_data_Ne_585_138.csv"
# file_name = "CT_data_SPDC_130.csv"
file = glob.glob("*{}*".format(file_name))[0]

data = pd.read_csv(file)

distance = []
ct = []
yerr = []
yerr1 = []
yerr2 = []


pix1 = 134

data_cut = data.loc[data["Pixel 1"] == pix1]

pix2 = data["Pixel 2"].unique()
pix2 = np.delete(pix2, np.where(pix2 <= pix1)[0])

for i, pix in enumerate(pix2):
    ct_pix = data_cut[data_cut["Pixel 2"] == pix].CT.values
    timestamps = data_cut[data_cut["Pixel 2"] == pix].Timestamps.values

    if ct_pix.size <= 0:
        continue

    distance.append(pix - pix1)
    if len(ct_pix > 1):
        ct.append(np.average(ct_pix))
    else:
        ct.append(ct_pix)
    a = np.mean(ct_pix)
    yerr1.append(np.sqrt(a * (1 - a) / np.mean(timestamps)))
    yerr2.append(sem(ct_pix))

plt.rcParams.update({"font.size": 20})
fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111)
ax1.errorbar(distance, ct, yerr=yerr2, color="salmon")
ax1.set_xlabel("Distance in pixels [-]")
ax1.set_ylabel("Average cross-talk [%]")
ax1.set_title("Pixel {}".format(pix1))

# ax2 = ax1.inset_axes([0.5, 0.5, 0.3, 0.45]) # position: x0, y0, width, height
# ax2.set_xticks((3.5, 4.5, 5.5, 6.5))
# # ax2.set_yticks((0, 10))
# # ax2.tick_params(axis='both', labelsize=16)
# ax2.set_xlim(3.5, 6.5)
# ax2.set_ylim(0.0045, 0.0562)
# ax2.errorbar(distance, ct, yerr, color='salmon')
# ax1.indicate_inset_zoom(ax2, edgecolor="black") # indicate zoom
