import glob
import os

import numpy as np

from functions import unpack as f_up

path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_703"
os.chdir(path)

files = glob.glob("*.dat*")

# data = f_up.unpack_2212(files[0], board_number="A5", fw_ver="block", timestamps=1000)


# =============================================================================
# numpy
# =============================================================================

filename = files[0]

timestamps = 1000

rawFile = np.fromfile(filename, dtype=np.uint32)

data_t = (rawFile & 0xFFFFFFF).astype(np.longlong) * 17.857
data_p = ((rawFile >> 28) & 0x3).astype(int)
data_t[np.where(rawFile < 0x80000000)] = -2
cycles = int(len(data_t) / timestamps / 65)

data_matrix_p = (
    data_p.reshape(cycles, 65, timestamps)
    .transpose((1, 0, 2))
    .reshape(65, timestamps * cycles)
)

data_matrix_t = (
    data_t.reshape(cycles, 65, timestamps)
    .transpose((1, 0, 2))
    .reshape(65, timestamps * cycles)
)

data_matrix_p = data_matrix_p[:-1]
data_matrix_t = data_matrix_t[:-1]

for i in range(64):
    data_matrix_p[i] += 4 * i

data_matrix_p1 = data_matrix_p.flatten()
data_matrix_t1 = data_matrix_t.flatten()
data_matrix_p1 = np.insert(
    data_matrix_p1,
    np.linspace(timestamps, cycles * timestamps, 1000).astype(int),
    np.arange(256),
)

data_out = {}

for i in range(255):
    data_out["{}".format(i)] = []

for key in data_out.keys():
    indices = np.argwhere(data_matrix_p1 == i)
    data_out[key].extend(data_matrix_t1[indices])
    data_out[key] = np.array(data_out[key])
    # data_out[key] = np.insert(
    #     data_out[key], np.linspace(timestamps, cycles * timestamps, 1000).astype(int), -2
    # )
