"""This script plots number of valid timestamp in each pixel using data from
three different setups: with sensor cover only (without optical fiber
attached), with cover and the fiber using a phone flashlight, and with cover,
fiber and Ne lamp.

"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up

path_cover = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "w cover wo fiber"

path_phone = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "phone w filter"

path_lamp = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "lamp w filter"

path_lamp_beeg = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/lamp w filter/beeg booi"

path_Ne_trig = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/Ne lamp ext trig"

path_Ne_trig2 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/Ne lamp ext trig/setup 2"

path_30 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "Ne lamp ext trig/30 lines of data"

# # =============================================================================
# # Data collected with the sensor cover but without the optical fiber attached
# # =============================================================================

# os.chdir(path_cover)

# DATA_FILES = glob.glob('*acq*'+'*dat*')

# valid_per_pixel = np.zeros(256)

# for i, num in enumerate(DATA_FILES):
#     data_matrix = f_up.unpack_binary_10(num)
#     for j in range(len(data_matrix)):
#         valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

#     plt.ioff()
#     plt.figure(figsize=(16, 10))
#     plt.rcParams.update({"font.size": 20})
#     plt.title("{}".format(num))
#     plt.xlabel("Pixel [-]")
#     plt.ylabel("Valid timestamps [-]")
#     plt.plot(valid_per_pixel, 'o')

#     try:
#         os.chdir("results")
#     except Exception:
#         os.mkdir("results")
#         os.chdir("results")

#     plt.savefig("{}.png".format(num))
#     os.chdir("..")

# # =============================================================================
# # Data collected with a phone flashlight using the optical fiber and the sensor
# # cover
# # =============================================================================

# os.chdir(path_phone)

# DATA_FILES = glob.glob('*acq*'+'*dat*')

# valid_per_pixel = np.zeros(256)

# for i, num in enumerate(DATA_FILES):
#     data_matrix = f_up.unpack_binary_10(num)
#     for j in range(len(data_matrix)):
#         valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

#     plt.ioff()
#     plt.figure(figsize=(16, 10))
#     plt.rcParams.update({"font.size": 20})
#     plt.title("{}".format(num))
#     plt.xlabel("Pixel [-]")
#     plt.ylabel("Valid timestamps [-]")
#     plt.plot(valid_per_pixel, 'o', color='salmon')

#     try:
#         os.chdir("results")
#     except Exception:
#         os.mkdir("results")
#         os.chdir("results")

#     plt.savefig("{}.png".format(num))
#     os.chdir("..")

# # =============================================================================
# # Data collected with the Ne lamp, sensor cover and optical fiber
# # =============================================================================

# os.chdir(path_lamp)

# DATA_FILES = glob.glob('*acq*'+'*dat*')

# valid_per_pixel = np.zeros(256)

# for i, num in enumerate(DATA_FILES):
#     data_matrix = f_up.unpack_binary_10(num)
#     for j in range(len(data_matrix)):
#         valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

#     plt.ioff()
#     plt.figure(figsize=(16, 10))
#     plt.rcParams.update({"font.size": 20})
#     plt.title("{}".format(num))
#     plt.xlabel("Pixel [-]")
#     plt.ylabel("Valid timestamps [-]")
#     plt.plot(valid_per_pixel, 'o', color='turquoise')

#     try:
#         os.chdir("results")
#     except Exception:
#         os.mkdir("results")
#         os.chdir("results")

#     plt.savefig("{}.png".format(num))
#     os.chdir("..")

# # =============================================================================
# # Big data files with Ne lamp, cover and optical fiber.
# # =============================================================================

# os.chdir(path_lamp_beeg)

# DATA_FILES = glob.glob('*acq*'+'*dat*')

# valid_per_pixel = np.zeros(256)

# for i, num in enumerate(DATA_FILES):
#     data_matrix = f_up.unpack_binary_512(num)
#     for j in range(len(data_matrix)):
#         valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

#     plt.ioff()
#     plt.figure(figsize=(16, 10))
#     plt.rcParams.update({"font.size": 20})
#     plt.title("{}".format(num))
#     plt.xlabel("Pixel [-]")
#     plt.ylabel("Valid timestamps [-]")
#     plt.plot(valid_per_pixel, 'o', color='orange')

#     try:
#         os.chdir("results")
#     except Exception:
#         os.mkdir("results")
#         os.chdir("results")

#     plt.savefig("{}.png".format(num))
#     os.chdir("..")

# =============================================================================
# External trigger
# =============================================================================

os.chdir(path_30)

DATA_FILES = glob.glob('*acq*'+'*dat*')

valid_per_pixel = np.zeros(256)

for i, num in enumerate(DATA_FILES):
    data_matrix = f_up.unpack_binary_flex(num, 30)
    for j in range(len(data_matrix)):
        valid_per_pixel[j] = len(np.where(data_matrix[j] > 0)[0])

    plt.ioff()
    plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 20})
    plt.title("{}".format(num))
    plt.xlabel("Pixel [-]")
    plt.ylabel("Valid timestamps [-]")
    plt.plot(valid_per_pixel, 'o', color='orange')

    try:
        os.chdir("results")
    except Exception:
        os.mkdir("results")
        os.chdir("results")

    plt.savefig("{}.png".format(num))
    os.chdir("..")
