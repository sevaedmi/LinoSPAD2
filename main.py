""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np
from functions import delta_t, plot_valid, fits as gf
from functions.calibrate import calibrate_save, calibrate_load
from functions import fits

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

path_v_585 = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585"

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

# delta_t.plot_grid_calib_mult(
#     path_v_585,
#     board_number="A5",
#     pix=(225, 226, 236, 237),
#     range_left=-20e3,
#     range_right=20e3,
#     timestamps=80,
#     same_y=False,
#     mult_files=False,
# )

delta_t.plot_grid_calib_mult_cut(
    path_v_585,
    board_number="A5",
    pix=(134, 135, 151, 152),
    range_left=-20e3,
    range_right=20e3,
    timestamps=80,
    same_y=False,
    mult_files=True,
)
