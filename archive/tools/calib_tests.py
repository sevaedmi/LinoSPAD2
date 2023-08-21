import glob
import os
import sys
from struct import unpack

import numpy as np

from functions import unpack as f_up
from functions.calibrate import calibrate_load

path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_585"
os.chdir(path)
file = glob.glob("*.dat*")[0]
timestamps = 50
board_number = "A5"
fw_ver = "block"

# ===============================================================================


timestamp_list = {}

for i in range(0, 256):
    timestamp_list["{}".format(i)] = []

# Variables that follow the cycle and the TDC numbers
cycler = 0
tdc = 0


# Function for assigning pixel addresses bases on the type of the 2212
# firmware version
def _pix_num(tdc_num, pix_coor):
    if fw_ver == "block":
        out = 4 * tdc_num + pix_coor
    elif fw_ver == "skip":
        out = tdc_num + 64 * pix_coor

    return out


with open(file, "rb") as f:
    while True:
        rawpacket = f.read(4)
        # All steps are in units of 32 bits
        # Reaching the end of a cycle, assign a '-1' to each pixel
        if not cycler % (32 * 65 * timestamps) and cycler != 0:
            for i in range(256):
                timestamp_list["{}".format(i)].append(-1)
        # Next TDC
        if not cycler % (32 * timestamps) and cycler != 0:
            tdc += 1
        cycler += 32
        # Cut the 64th (TDC=[0,...63]) that does not contain timestamps
        if tdc != 0 and tdc == 64:
            continue
        # Reset the TDC number - end of cycle
        if tdc != 0 and tdc == 65:
            tdc = 0
        if not rawpacket:
            break
        packet = unpack("<I", rawpacket)
        if (packet[0] >> 31) == 1:
            pix_coor = (packet[0] >> 28) & 0x3
            address = _pix_num(tdc, pix_coor)
            timestamp_list["{}".format(address)].append((packet[0] & 0xFFFFFFF))

for key in timestamp_list:
    timestamp_list[key] = np.array(timestamp_list[key]).astype(np.longlong)

# path to the current script, two levels up (the script itself is in the path) and
# one level down to the calibration data
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
    ind = np.where(np.array(timestamp_list["{}".format(i)]) >= 0)[0]
    # if not np.any(ind):
    #     continue
    timestamp_list["{}".format(i)][ind] = (
        timestamp_list["{}".format(i)][ind] - timestamp_list["{}".format(i)][ind] % 140
    ) * 17.857 + cal_mat[i, (timestamp_list["{}".format(i)][ind] % 140)]

# ===============================================================================


data = f_up.unpack_2212(file, board_number="A5", fw_ver="block", timestamps=50)

valid_per_pix = np.zeros(256)
for i in range(256):
    valid_per_pix[i] = len(np.where(data["{}".format(i)] >= 0)[0])

valid_all = np.sum(valid_per_pix)

# =============================================================================
#
# =============================================================================


def unpack_2212__(filename, board_number, fw_ver, timestamps: int = 512):
    timestamp_list = {}

    for i in range(0, 256):
        timestamp_list["{}".format(i)] = []

    # Variables that follow the cycle and the TDC numbers
    cycler = 0
    tdc = 0

    # Function for assigning pixel addresses bases on the type of the 2212
    # firmware version
    def _pix_num(tdc_num, pix_coor):
        if fw_ver == "block":
            out = 4 * tdc_num + pix_coor
        elif fw_ver == "skip":
            out = tdc_num + 64 * pix_coor

        return out

    with open(filename, "rb") as f:
        while True:
            rawpacket = f.read(4)
            # All steps are in units of 32 bits
            # Reaching the end of a cycle, assign a '-1' to each pixel
            if not cycler % (32 * 65 * timestamps) and cycler != 0:
                for i in range(256):
                    timestamp_list["{}".format(i)].append(-1)
            # Next TDC
            if not cycler % (32 * timestamps) and cycler != 0:
                tdc += 1
            cycler += 32
            # Cut the 64th (TDC=[0,...63]) that does not contain timestamps
            if tdc != 0 and tdc == 64:
                continue
            # Reset the TDC number - end of cycle
            if tdc != 0 and tdc == 65:
                tdc = 0
            if not rawpacket:
                break
            packet = unpack("<I", rawpacket)
            if (packet[0] >> 31) == 1:
                pix_coor = (packet[0] >> 28) & 0x3
                address = _pix_num(tdc, pix_coor)
                timestamp_list["{}".format(address)].append(
                    (packet[0] & 0xFFFFFFF) * 17.857
                )

    for key in timestamp_list:
        timestamp_list[key] = np.array(timestamp_list[key])

    return timestamp_list


data1 = unpack_2212__(file, board_number="A5", fw_ver="block", timestamps=50)

valid_per_pix1 = np.zeros(256)
for i in range(256):
    valid_per_pix1[i] = len(np.where(data1["{}".format(i)] >= 0)[0])

valid_all1 = np.sum(valid_per_pix1)
