from functions import unpack as f_up
import os
import glob
import numpy as np
from matplotlib import pyplot as plt

path_back = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "Ne lamp ext trig/dry run - background"

os.chdir(path_back)
file_back = glob.glob('*.dat*')[0]

path_lamp = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "Ne lamp ext trig"

os.chdir(path_lamp)
file_lamp = glob.glob('*.dat*')[0]

os.chdir(path_back)
data_back = f_up.unpack_binary_512(file_back)

os.chdir(path_lamp)
data_lamp = f_up.unpack_binary_512(file_lamp)

valid_back = np.zeros(256)
valid_lamp = np.zeros(256)

for j in range(len(data_back)):
    valid_back[j] = len(np.where(data_back[j] > 0)[0])

for j in range(len(data_lamp)):
    valid_lamp[j] = len(np.where(data_lamp[j] > 0)[0])

valid_back = np.delete(valid_back, (15, 93, 236))
valid_lamp = np.delete(valid_lamp, (15, 93, 236))


plt.ion()
plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 20})
plt.xlabel("Pixel [-]")
plt.ylabel("Valid timestamps [-]")
plt.plot(valid_back, 'o', color='orange')
plt.plot(valid_lamp, 'o')
