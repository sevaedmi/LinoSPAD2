""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np
from functions import delta_t, plot_valid, fits as gf
from functions.calibrate import calibrate_save, calibrate_load
from functions import fits
from tools.collect_ct import collect_ct as cct

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

# =========================================================================
# Function execution.
# =========================================================================

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

# fits.fit_gauss_cal_mult(
#     path_v_585,
#     pix=(225, 226, 236, 237),
#     board_number="A5",
#     range_left=-15e3,
#     range_right=15e3,
#     timestamps=80,
#     mult_files=True,
# )


# fits.fit_gauss_cal_mult_cut(
#     path_v_585,
#     pix=(225, 226, 236, 237),
#     board_number="A5",
#     timestamps=80,
#     mult_files=True,
# )


# plot_valid.plot_calib_mult(
#     path_v_585, board_number="A5", timestamps=80, mask=mask, mult_files=True
# )

delta_t.plot_grid_mult_cut(
    path_BNL,
    board_number="A5",
    pix=(137, 138, 151, 152),
    range_left=-10e3,
    range_right=10e3,
    timestamps=80,
    same_y=False,
)

# delta_t.plot_grid_mult_2212(
#     path_FW2212,
#     board_number="A5",
#     fw_ver = "block",
#     pix=(137, 138, 151, 152),
#     range_left=-10e3,
#     range_right=10e3,
#     timestamps=160,
#     same_y=False,
# )

# plot_valid.plot_valid_mult(path_BNL, timestamps=80, board_number="A5")

# plot_valid.plot_valid_FW2212_mult(
#     path_FW2212, board_number="A5", fw_ver="block", timestamps=160
# )
