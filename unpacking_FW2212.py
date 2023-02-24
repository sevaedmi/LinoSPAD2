import os
from glob import glob
import numpy as np
from functions.calibrate import calibrate_load
from matplotlib import pyplot as plt
import sys

# path = "D:/LinoSPAD2/Data/board_A5/FW 2212 skip"
path = "D:/LinoSPAD2/Data/board_A5/FW 2212 block"

board_number = "A5"

os.chdir(path)

filename = glob("*.dat*")[0]

timestamps = 1536

rawFile = np.fromfile(filename, dtype=np.uint32)
address = ((rawFile >> 28) & 0x3).astype(np.int64)
data = (rawFile & 0xFFFFFFF).astype(np.longlong)
data[np.where(rawFile < 0x80000000)] = -1


def pix_num(tdc_num, pix_coor):
    if "block" in path:
        out = 4 * tdc_num + pix_coor - 3
    elif "skip" in path:
        out = tdc_num + 63 * pix_coor + pix_coor

    return out


output = {}

for i in range(1, 257):
    output["{}".format(i)] = []
tdc_num = 1
cycle = 1

for i in range(1, len(data) + 1):
    if i % timestamps == 0:
        tdc_num += 1
    if i % (1536 * 65) == 0:
        cycle += 1
        tdc_num = 1
    if tdc_num % 65 == 0:
        continue
    pix_add = pix_num(tdc_num, address[i - 1])

    output["{}".format(pix_add)].append(data[i - 1])

# # Calibration
# path_calib_data = os.path.realpath(__file__) + "/.." + "/calibration_data"

# try:
#     cal_mat = calibrate_load(path_calib_data, board_number)
# except FileNotFoundError:
#     print(
#         "No .csv file with the calibration data was found, check the path "
#         "or run the calibration."
#     )
#     sys.exit()

# for i in range(1, 257):
#     a = np.array(output["{}".format(i)])
#     if not any(a):
#         continue
#     ind = np.where(a >= 0)[0]
#     a[ind] = (a[ind] - a[ind] % 140) * 17.857 + cal_mat[i, (a[ind] % 140)]

valid_per_pix = np.zeros(256)

for i in range(1, 257):
    a = np.array(output["{}".format(i)])
    valid_per_pix[i - 1] = len(np.where(a > 0)[0])

plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(16, 10))
plt.plot(valid_per_pix, "-o", color="salmon")
plt.xlabel("Pixel number [-]")
plt.ylabel("Timestamps [-]")

try:
    os.chdir("results")
except FileNotFoundError:
    os.mkdir("results")
    os.chdir("results")
fig.tight_layout()  # for perfect spacing between the plots
plt.savefig("{}.png".format(filename))
os.chdir("../..")
