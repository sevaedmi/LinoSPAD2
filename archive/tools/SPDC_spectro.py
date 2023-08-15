import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import unpack

path = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer"
path_bckg = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/bckg"

os.chdir(path)

files = glob.glob("*.dat*")

valid_per_pixel = np.zeros(256)

pix_coor = np.arange(256).reshape(64, 4)

for i, file in enumerate(files):
    data_all = unpack.unpack_2212_numpy(file, board_number="A5", timestamps=10)

    for i in np.arange(0, 256):
        tdc, pix = np.argwhere(pix_coor == i)[0]
        ind = np.where(data_all[tdc].T[0] == pix)[0]
        ind1 = ind[np.where(data_all[tdc].T[1][ind] > 0)[0]]
        valid_per_pixel[i] += len(data_all[tdc].T[1][ind1])

os.chdir(path_bckg)

files = glob.glob("*.dat*")

valid_per_pixel_bckg = np.zeros(256)

for i, file in enumerate(files):
    data_all_bckg = unpack.unpack_2212_numpy(file, board_number="A5", timestamps=10)

    for i in np.arange(0, 256):
        tdc, pix = np.argwhere(pix_coor == i)[0]
        ind = np.where(data_all_bckg[tdc].T[0] == pix)[0]
        ind1 = ind[np.where(data_all_bckg[tdc].T[1][ind] > 0)[0]]
        valid_per_pixel_bckg[i] += len(data_all_bckg[tdc].T[1][ind1])

mask = [57, 70, 95, 123, 157, 165, 187, 205, 212]
for i in mask:
    valid_per_pixel[i] = 0

plt.ion()
plt.figure(figsize=(16, 10))
plt.xlabel("Pixel [-]")
plt.ylabel("Timestamps [-]")
# plt.plot(valid_per_pixel, "o-", color="salmon")
# plt.yscale("log")
# plt.plot(valid_per_pixel_bckg, 'o-', color='dimgray')
plt.plot(valid_per_pixel - valid_per_pixel_bckg, "o-", color="teal")
plt.ylim(-1e3)
plt.show()

# =============================================================================
# Anti correlation
# =============================================================================

pix_coor = np.arange(256).reshape(64, 4)

pix_left = np.arange(94, 157, 1)
pix_right = np.arange(185, 245, 1)

mat = np.zeros((256, 256))

delta_window = 15e3

for i in tqdm(range(len(files)), desc="Anticorrelation plot, going through files"):
    deltas_all = {}

    # Unpack data for the requested pixels into dictionary
    data_all = unpack.unpack_2212_numpy(files[i], board_number="A5", timestamps=20)

    # Calculate and collect timestamp differences
    for q in pix_left:
        for w in pix_right:
            deltas_all["{},{}".format(q, w)] = []

            # find end of cycles
            cycler = np.argwhere(data_all[0].T[0] == -2)
            # TODO: most probably losing first delta t due to cycling
            cycler = np.insert(cycler, 0, 0)
            # first pixel in the pair
            tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
            pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
            # second pixel in the pair
            tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
            pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
            # get timestamp for both pixels in the given cycle
            for cyc in np.arange(len(cycler) - 1):
                pix1_ = pix1[np.logical_and(pix1 > cycler[cyc], pix1 < cycler[cyc + 1])]
                if not np.any(pix1_):
                    continue
                pix2_ = pix2[np.logical_and(pix2 > cycler[cyc], pix2 < cycler[cyc + 1])]
                if not np.any(pix2_):
                    continue
                # calculate delta t
                tmsp1 = data_all[tdc1].T[1][
                    pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                ]
                tmsp2 = data_all[tdc2].T[1][
                    pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                ]
                for t1 in tmsp1:
                    deltas = tmsp2 - t1
                    ind = np.where(np.abs(deltas) < delta_window)[0]
                    deltas_all["{},{}".format(q, w)].extend(deltas[ind])
                    mat[q][w] += len(ind)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))
plt.xlabel("Pixel [-]")
plt.ylabel("Pixel [-]")
# plt.xlim(93, 158)
# plt.ylim(184, 246)
pos = ax.imshow(mat.T, cmap="cividis", interpolation="none", origin="lower")
# pos = ax.matshow(mat, cmap="Blues", origin='lower')
fig.colorbar(pos, ax=ax)
