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

# runs the applicaiton
run_application = True

if run_application is True:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
else:

    tools.jakub_playground.main_playground()

    # =========================================================================
    # Paths to where either data or the 'csv' files with the resuts are located.
    # =========================================================================
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



