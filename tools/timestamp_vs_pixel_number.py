"""Script for plotting a 2D plot of timestamps vs pixel number.
"""
from functions import unpack as f_up
import glob
import os
from matplotlib import pyplot as plt
import numpy as np

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "Ne lamp ext trig/setup 2/3 ms acq window"

os.chdir(path)

filename = glob.glob('*.dat*')[0]

data = f_up.unpack_binary_512(filename)

range_x = np.arange(0, 255, 1)
range_y = np.arange(0, 3.9e9, 1e6)

plt.figure(figsize=(16, 10))
plt.xlabel('Pixel number [-]')
plt.ylabel('Timestamps [ps]')
plt.plot(data, '.')
