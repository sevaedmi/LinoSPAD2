"""Script for quick plotting of number of valid timestamps per pixel.

"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "Ar lamp/FW 2208"

os.chdir(path)

filename = glob.glob("*.dat*")[-1]

data = f_up.unpack_binary_flex(filename, 512)

row_valid = np.zeros(len(data))

for i, row in enumerate(data):
    row_valid[i] = len(np.where(row > -1)[0])

plt.ion()

plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 22})
plt.xlabel("Pixel number [-]")
plt.ylabel("Number of valid timestamps [-]")
plt.yscale('log')
# plt.ylim(0, row_valid[159]+100)
plt.plot(row_valid, 'o', color='salmon')
