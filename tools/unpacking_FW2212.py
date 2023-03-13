import os
import sys
from glob import glob
from struct import unpack

import numpy as np
from matplotlib import pyplot as plt

from functions import unpack as f_up
from functions.calibrate import calibrate_load

# path = "D:/LinoSPAD2/Data/board_A5/FW 2212 skip"
# path = "D:/LinoSPAD2/Data/board_A5/FW 2212 block"
path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_585"
# path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_skip"


board_number = "A5"

timestamps = 50

os.chdir(path)

filename = glob("*.dat*")[0]

# =============================================================================
# The fast way
# =============================================================================

rawFile = np.fromfile(filename, dtype=np.uint32)
address = ((rawFile >> 28) & 0x3).astype(np.int64)
data = (rawFile & 0xFFFFFFF).astype(np.longlong)
data[np.where(rawFile < 0x80000000)] = -1

# =============================================================================
# Notation with indeces starting from 0: block version does not work
# =============================================================================


def pix_num(tdc_num, pix_coor):
    if "block" in path:
        out = 4 * tdc_num + pix_coor
    elif "skip" in path:
        out = tdc_num + 64 * pix_coor

    return out


output = {}

for i in range(0, 256):
    output["{}".format(i)] = []
tdc_num = 0

for i in range(0, len(data)):
    if i != 0 and i % timestamps == 0:
        tdc_num += 1
    if tdc_num != 0 and tdc_num == 64:
        continue
    if tdc_num != 0 and tdc_num == 65:
        tdc_num = 0
    # if i % (timestamps+65) == 0:
    pix_add = pix_num(tdc_num, address[i])
    output["{}".format(pix_add)].append(data[i])

for key in output:
    output[key] = np.array(output[key])


path_calib_data = "C:/Users/bruce/Documents/GitHub/LinoSPAD2/calibration_data"

try:
    cal_mat = calibrate_load(path_calib_data, board_number)
except FileNotFoundError:
    print(
        "No .csv file with the calibration data was found, check the path "
        "or run the calibration."
    )
    sys.exit()
for i in range(256):
    ind = np.where(np.array(output["{}".format(i)]) >= 0)[0]
    if not np.any(ind):
        continue
    output["{}".format(i)][ind] = (
        output["{}".format(i)][ind] - output["{}".format(i)][ind] % 140
    ) * 17.857 + cal_mat[i, (output["{}".format(i)][ind] % 140)]


valid_per_pix = np.zeros(256)

for i in range(0, 256):
    a = np.array(output["{}".format(i)])
    valid_per_pix[i] = len(np.where(a > 0)[0])

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

valid_per_pix[mask] = 0

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
fig.tight_layout()
plt.savefig("{}.png".format(filename))
os.chdir("..")


# =============================================================================
# The BS way
# =============================================================================

output = {}

for i in range(0, 256):
    output["{}".format(i)] = []

cycler = 0
tdc = 0


def pix_num(tdc_num, pix_coor):
    if "block" in path:
        out = 4 * tdc_num + pix_coor
    elif "skip" in path:
        out = tdc_num + 64 * pix_coor

    return out


with open(filename, "rb") as f:
    while True:
        rawpacket = f.read(4)

        if not cycler % (32 * 65 * timestamps) and cycler != 0:
            for i in range(256):
                output["{}".format(i)].append(-1)
        if not cycler % (32 * timestamps) and cycler != 0:
            tdc += 1
        cycler += 32
        if tdc != 0 and tdc == 64:
            continue
        if tdc != 0 and tdc == 65:
            tdc = 0
        if not rawpacket:
            break
        packet = unpack("<I", rawpacket)
        if (packet[0] >> 31) == 1:
            pix_coor = (packet[0] >> 28) & 0x3
            address = pix_num(tdc, pix_coor)
            output["{}".format(address)].append(
                # (packet[0] & 0xFFFFFFF)*17.857
                (packet[0] & 0xFFFFFFF)
            )

for key in output:
    output[key] = np.array(output[key])


path_calib_data = "C:/Users/bruce/Documents/GitHub/LinoSPAD2/calibration_data"

try:
    cal_mat = calibrate_load(path_calib_data, board_number)
except FileNotFoundError:
    print(
        "No .csv file with the calibration data was found, check the path "
        "or run the calibration."
    )
    sys.exit()
for i in range(256):
    ind = np.where(np.array(output["{}".format(i)]) >= 0)[0]
    if not np.any(ind):
        continue
    output["{}".format(i)][ind] = (
        output["{}".format(i)][ind] - output["{}".format(i)][ind] % 140
    ) * 17.857 + cal_mat[i, (output["{}".format(i)][ind] % 140)]


valid_per_pix = np.zeros(256)

for i in range(0, 256):
    a = np.array(output["{}".format(i)])
    valid_per_pix[i] = len(np.where(a > 0)[0])

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

valid_per_pix[mask] = 0

plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(16, 10))
plt.plot(valid_per_pix, "-o", color="salmon")
plt.xlabel("Pixel number [-]")
plt.ylabel("Timestamps [-]")

# =======================================
data11 = f_up.unpack_2212(filename, board_number="A5", fw_ver="block", timestamps=80)

valid_per_pix = np.zeros(256)

for i in range(0, 256):
    a = np.array(data11["{}".format(i)])
    valid_per_pix[i] = len(np.where(a > 0)[0])

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

valid_per_pix[mask] = 0

plt.rcParams.update({"font.size": 22})
fig = plt.figure(figsize=(16, 10))
plt.plot(valid_per_pix, "-o", color="salmon")
plt.xlabel("Pixel number [-]")
plt.ylabel("Timestamps [-]")
