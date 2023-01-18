""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np
from functions import delta_t, plot_valid, fits as gf
from functions.calibrate import calibrate_save, calibrate_load
from functions.fits import fit_gauss_cal

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

path_v = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
    "Data/board_A5/V_setup"
)

path_v_585 = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
    "Data/board_A5/V_setup/Ne_585"
)

# =========================================================================
# Function execution.
# =========================================================================

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

# delta_t.plot_grid_calib(
#     path_v_585,
#     board_number="A5",
#     pix=(9, 10, 11, 48, 49, 50, 51),
#     range_left=-25e3,
#     range_right=25e3,
#     timestamps=80,
#     same_y=False,
# )

# fit_gauss_cal(
#     path_v_585,
#     pix=(9, 10, 11, 48, 49, 50, 51),
#     board_number="A5",
#     range_left=-5e3,
#     range_right=5e3,
#     timestamps=80,
# )

# plot_valid.plot_calib(
#     path_v_585,
#     mask=mask,
#     pix=[9, 10, 11, 48, 49, 50, 51],
#     board_number="A5",
#     timestamps=80,
# )

delta_t.plot_grid_calib_mult(
    path_v_585,
    board_number="A5",
    pix=(9, 10, 11, 48, 49, 50, 51),
    range_left=-15e3,
    range_right=15e3,
    timestamps=80,
    same_y=False,
)
