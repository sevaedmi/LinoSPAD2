""" The main hub for analysis of data from LinoSPAD2.

"""
# Scripts imports
import numpy as np
from functions import delta_t, plot_valid, fits as gf

# Application imports
import sys
from app.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import tools.jakub_playground

run_application = False


if run_application is True:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
else:
    # =========================================================================
    # Paths to where either data or the 'csv' files with the resuts are located.
    # =========================================================================

    tools.jakub_playground.main_playground()

    # First peak 84 85 86 87 88 89
    # Second peak 218 219 220 221 222

    # path_2208_540 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD"\
    #     "/Software/Data/Ne lamp/FW 2208/540 nm"
    #
    # path_2208_653 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD"\
    #     "/Software/Data/Ne lamp/FW 2208/653 nm"
    #
    # path_2208_Ar = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    #     "Software/Data/Ar lamp/FW 2208"
    #
    # path_2208_TF = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    #     "Software/Data/Ne lamp/FW 2208/two fiber"

    # =========================================================================
    # Function execution.
    # =========================================================================

    # pix = np.arange(140, 161, 1)
    # plot_valid.online_plot_valid(path_2208_653, pix_range=pix)

    # pix = np.arange(130, 161, 1)
    # plot_valid.plot_valid(path_2208_653, pix=pix, timestamps=512,
    #                       scale='log', show_fig=False)

    # delta_t.plot_grid(path_2208_653, (135, 136, 137, 138, 140), show_fig=False,
    #                   same_y=True)
