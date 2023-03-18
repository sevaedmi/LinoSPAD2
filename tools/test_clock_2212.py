""" Testing the internal clock settings of the LinoSPAD2 firmware
version 2212/2208.
"""

import glob
import os

import numpy as np

from functions import unpack as f_up

path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_700"
os.chdir(path)

file = glob.glob("*.dat*")[0]

data = f_up.unpack_numpy(file, board_number="A5", timestamps=512)

for i in range(len(data)):
    for j in range(0, len(data[i]) - 1):
        check = 0
        if data[i][j] > check:
            check = data[i][j]
        elif data[i][j + 1] < check and data[i][j + 1] != -1:
            print(i, j)
            check = 0
            break
        else:
            continue
