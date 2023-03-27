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

path_585 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_585"

path_694 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_694"

path_700 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/Ne_700"

path_SPDC = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2208/SPDC"

path_585_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_585"

path_694_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_694"

path_694_12_temp = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_694/temp"

path_700_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_700"

path_SPDC_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/SPDC"

# =========================================================================
# Function execution.
# =========================================================================

# mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

delta_t.deltas_save(
    path_700_12, pix=(3, 45), board_number="A5", timestamps=1000, fw_ver="2212b"
)

delta_t.delta_cp(
    path_700_12,
    pix=(3, 45, 100),
    rewrite=True,
    range_left=-10e3,
    range_right=10e3,
)

# gf.fit_wg(path_700, pix_pair=(3, 51), window=10e3)

plot_valid.plot_valid_FW2212_mult(
    path_700_12, board_number="A5", fw_ver="block", timestamps=1000, show_fig=True
)
