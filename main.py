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

    # tools.jakub_playground.main_playground()

    # First peak 51 - 58
    # Second peak 203 - 211

    # path_2208_540 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD"\
    #     "/Software/Data/Ne lamp/FW 2208/540 nm"
    #
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

    # =========================================================================
    # Function execution.
    # =========================================================================

    # pix = np.arange(140, 161, 1)
    # plot_valid.online_plot_valid(path_2208_653, pix_range=pix)

    pix = np.arange(130, 161, 1)
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
    # plot_valid.plot_valid(
    #     path_BNL,
    #     pix=pix,
    #     mask=mask,
    #     timestamps=512,
    #     scale="log",
    #     style="o-",
    #     show_fig=True,
    # )

    delta_t.plot_grid(path_BNL, (87, 88, 223), show_fig=True, same_y=True) 

    # %load_ext line_profiler
    # %lprun -f delta_t.plot_grid(path_BNL, (87, 88, 223), show_fig=True, same_y=True)
# =============================================================================
# Profiler: evaluation of code performance in terms of time consumption, line by line
# =============================================================================

# from line_profiler import LineProfiler

# lp = LineProfiler()
# lp_wrapper = lp(delta_t.plot_grid(path_BNL, (87, 88, 223), show_fig=True,
#                                   same_y=True))
# lp.print_stats()
