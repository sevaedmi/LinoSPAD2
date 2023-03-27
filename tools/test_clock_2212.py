""" Testing the internal clock settings of the LinoSPAD2 firmware
version 2212/2208.
"""

import glob
import os

import numpy as np

from functions import unpack as f_up

path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_700"

os.chdir(path)

file = glob.glob("*.dat*")[0]

data = f_up.unpack_2212(file, board_number="A5", fw_ver="block", timestamps=1000)

for i in range(len(data)):
    for j in range(0, len(data["{}".format(i)]) - 1):
        check = 0
        if data["{}".format(i)][j] > check:
            check = data["{}".format(i)][j]
        elif data["{}".format(i)][j + 1] < check and data["{}".format(i)][j + 1] != -2:
            print(i, j)
            break
        else:
            continue
