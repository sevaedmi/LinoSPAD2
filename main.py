""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np

from functions import cross_talk as ct
from functions import delta_t
from functions import fits as gf
from functions import plot_valid

# =========================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =========================================================================

path_test = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585/for_tests"

path_FW2212 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_585"

path_BNL = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585"

path_BNL_temp = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585/temp"

path_SPDC = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/SPDC"

# =========================================================================
# Function execution.
# =========================================================================

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

# delta_t.plot_grid_mult(
#     path_SPDC,
#     board_number="A5",
#     rewrite=False,
#     pix=(136, 137, 138, 139, 140, 154, 155, 156, 157, 158),
#     range_left=-20e3,
#     range_right=20e3,
#     timestamps=100
# )


delta_t.deltas_save(
    path_SPDC, pix=np.arange(134, 166, 1), board_number="A5", timestamps=100
)

# delta_t.deltas_save(
#     path_BNL_temp, pix=np.arange(134, 159, 1), board_number="A5", timestamps=40
# )

delta_t.delta_cp(
    path_BNL,
    pix=np.arange(135, 159, 1),
    rewrite=True,
    range_left=-20e3,
    range_right=20e3,
)

delta_t.delta_cp(
    path_SPDC,
    pix=np.arange(135, 159, 1),
    rewrite=True,
    range_left=-20e3,
    range_right=20e3,
)

# plot_valid.plot_valid_mult(path_test, timestamps=80, board_number="A5", show_fig=True)

# plot_valid.plot_valid_FW2212_mult(
#     path_FW2212, board_number="A5", fw_ver="block", timestamps=80
# )
