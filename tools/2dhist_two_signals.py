import os
from glob import glob
import numpy as np
from functions.unpack import unpack_calib as un
from matplotlib import pyplot as plt
from matplotlib import colors

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
    "Data/board_A5/V_setup/Ne_585"
)

os.chdir(path)

file = glob("*.dat*")[0]

data = un(file, timestamps=80, board_number="A5")

data225 = data[225]
data236 = data[236]

fig, ax = plt.subplots(figsize=(10, 7))
plt.rcParams.update({"font.size": 22})
ax.set_xlabel("Time, 1st signal [ms]")
ax.set_ylabel("Time, 2nd signal [ms]")
hist = ax.hist2d(x=data225, y=data236, bins=80, norm=colors.LogNorm())
clb = plt.colorbar(hist[3], ax=ax)
clb.ax.set_ylabel("Timestamps [-]", fontsize=15)

try:
    os.chdir("results")
except Exception:
    os.mkdir("results")
    os.chdir("results")
plt.savefig("{}_2dhist.png".format(file))
os.chdir("..")
