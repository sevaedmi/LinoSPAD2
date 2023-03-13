""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np

from functions import cross_talk as ct
from functions import delta_t
from functions import fits
from functions import fits as gf
from functions import plot_valid
from functions.calibrate import calibrate_load, calibrate_save

### The app has been moved to a standalone repo ###

# Application imports
# import sys
# from app.main_window import MainWindow
# from PyQt5.QtWidgets import QApplication

# # runs the applicaiton
# run_application = True

# if run_application is True:
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     app.exec()
# else:

# =========================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =========================================================================

path_test = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585/for_tests"

path_v_585 = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585"

path_FW2212 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_585"

path_BNL = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585"
path_temp = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585/temp"

path_SPDC = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585/SPDC"

path_ct = "C:/Users/bruce/Documents/Quantum astrometry"

# =========================================================================
# Function execution.
# =========================================================================

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

# delta_t.plot_grid_mult(
#     path_SPDC,
#     board_number="A5",
#     pix=(136, 137, 138, 139, 140, 154, 155, 156, 157, 158),
#     range_left=-20e3,
#     range_right=20e3,
#     timestamps=200
# )

delta_t.plot_grid_mult(
    path_BNL,
    board_number="A5",
    pix=(137, 138, 155, 156),
    range_left=-20e3,
    range_right=20e3,
    timestamps=100,
)

# delta_t.plot_grid_mult_2212(
#     path_FW2212,11112[-]
#     board_number="A5",
#     fw_ver="block",
#     pix=(137, 138, 155, 156),
#     range_left=-20e3,
#     range_right=20e3,
#     timestamps=80
# )

# plot_valid.plot_valid_mult(path_temp, timestamps=40, board_number="A5")

plot_valid.plot_valid_mult_nc(path_BNL, board_number="A5", timestamps=100)

# plot_valid.plot_valid_FW2212_mult(
#     path_FW2212, board_number="A5", fw_ver="block", timestamps=80
# )

# ct.collect_ct(path_SPDC, pix=np.arange(130, 165, 1), board_number="A5")
# ct.plot_ct(path_ct, pix1=138, scale='log')
