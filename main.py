""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np
from functions import delta_t, plot_valid, fits as gf
from functions.calibrate import calibrate_save, calibrate_load

# Application imports
import sys
from app.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import tools.jakub_playground

# runs the applicaiton
run_application = False

if run_application is True:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
else:

    # tools.jakub_playground.main_playground()

    # =========================================================================
    # Paths to where either data or the 'csv' files with the resuts are located.
    # =========================================================================
    path_2208_540 = (
        "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD"
        "/Software/Data/Ne lamp/FW 2208/540 nm"
    )

    path_2208_653 = (
        "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD"
        "/Software/Data/Ne lamp/FW 2208/653 nm"
    )
    #
    # path_2208_Ar = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    #     "Software/Data/Ar lamp/FW 2208"
    #
    # path_2208_TF = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    #     "Software/Data/Ne lamp/FW 2208/two fiber"

    path_BNL = (
        "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"
        "/Data/BNL-Jakub/SPDC"
    )

    path_cal = (
        "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
        "Data/calibration_data"
    )

    path_test = (
        "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
        "Data/board_A5/test"
    )

    path_calib_A5 = (
        "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"
        "Data/board_A5/calibration_data"
    )

    # =========================================================================
    # Function execution.
    # =========================================================================

    # pix = np.arange(140, 161, 1)
    # plot_valid.online_plot_valid(path_2208_653, pix_range=pix)

    # pix = np.arange(130, 161, 1)
    # warm/hot pixels, which are stable for this LinoSPAD2 desk
    mask = [
        15,
        16,
        29,
        39,
        40,
        50,
        52,
        66,
        73,
        93,
        95,
        96,
        98,
        101,
        109,
        122,
        127,
        196,
        210,
        231,
        236,
        238,
    ]

    # plot_valid.plot_valid(path_test, pix=(np.arange(140,160,1)), mask=[], timestamps=512)
    # delta_t.plot_grid(
    #     path_test, pix=(116, 117, 118, 119, 120), range_left=-5e3, range_right=5e3
    # )
    delta_t.plot_grid_calib(
        path_test, pix=(116, 117, 118, 119, 120), range_left=-5e3, range_right=5e3
    )
