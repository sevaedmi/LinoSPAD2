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
# run_application = False

# if run_application is True:
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     app.exec()
# else:

# =========================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =========================================================================

path_test = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
    "Data/board_A5/V_setup"
)

path_calib_A5 = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
    "Data/board_A5/calibration_data"
)

# =========================================================================
# Function execution.
# =========================================================================

# plot_valid.plot_valid(path_test, pix=(np.arange(0, 60, 1)), mask=mask,
#                       timestamps=512, show_fig=True)

# delta_t.plot_grid_calib(
#     path_test, board_number="A5", pix=(16, 17), range_left=-15e3,
#     range_right=15e3
# )

# fit_gauss_cal(
#     path_test, pix=(40, 41), board_number="A5", range_left=-5e3, range_right=5e3
# )

plot_valid.plot_calib(path_test, pix=[10, 11], board_number="A5")
