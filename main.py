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

path_test = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Ne/asdf"

path_123 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/release_2212/release_2212/shared/data"

path_NL11_noise_check = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/noise_check"

path_NL11_703_12 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/Ne_703"

path_NL11_703 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2208"

path_A5_SPDC_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer"

path_A5_Ar_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/Ar"

path_A5_703_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_703"

path_A5_693_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_693"

# =========================================================================
# Function execution.
# =========================================================================

# mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

# delta_t.deltas_save(
#     path_A5_703_12,
#     pix=(3, 45),
#     rewrite=True,
#     board_number="A5",
#     timestamps=1000,
#     fw_ver="2212b",
# )

delta_t.deltas_save_numpy(
    path_NL11_703_12,
    pix=(9, 220),
    rewrite=True,
    board_number="NL11",
    timestamps=140,
    fw_ver="2212b",
)

delta_t.delta_cp(
    path_NL11_703_12,
    pix=(9, 220),
    rewrite=True,
    range_left=-20e3,
    range_right=20e3,
)

gf.fit_wg(path_NL11_703_12, pix_pair=(9, 220), window=4e3)

# plot_valid.plot_valid_FW2212_mult(
#     path_NL11_703_12, board_number="NL11", fw_ver="block", timestamps=140, show_fig=True
# )
