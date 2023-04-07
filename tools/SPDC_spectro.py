import glob
import os

import numpy as np
from matplotlib import pyplot as plt

from functions import unpack

path = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer"
path_bckg = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/bckg"

os.chdir(path)

files = glob.glob("*.dat*")

valid_per_pixel = np.zeros(256)

for i, file in enumerate(files):
    data = unpack.unpack_2212(file, board_number="A5", fw_ver="block", timestamps=50)

    for i in range(0, 256):
        a = np.array(data["{}".format(i)])
        valid_per_pixel[i] = valid_per_pixel[i] + len(np.where(a > 0)[0])

os.chdir(path_bckg)

files = glob.glob("*.dat*")

valid_per_pixel_bckg = np.zeros(256)

for i, file in enumerate(files):
    data = unpack.unpack_2212(file, board_number="A5", fw_ver="block", timestamps=50)

    for i in range(0, 256):
        a = np.array(data["{}".format(i)])
        valid_per_pixel_bckg[i] = valid_per_pixel_bckg[i] + len(np.where(a > 0)[0])

mask = [57, 70, 95, 123, 157, 165, 187, 205, 212]
for i in mask:
    valid_per_pixel[i] = 0

plt.ion()
plt.figure(figsize=(16, 10))
plt.xlabel("Pixel [-]")
plt.ylabel("Timestamps [-]")
plt.plot(valid_per_pixel, "o-", color="salmon")
plt.yscale("log")
# plt.plot(valid_per_pixel_bckg, 'o-', color='dimgray')
# plt.plot(valid_per_pixel - valid_per_pixel_bckg, 'o-', color='teal')
plt.show()
