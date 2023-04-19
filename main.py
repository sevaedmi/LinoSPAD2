""" The main hub for analysis of data from LinoSPAD2.

"""

# Scripts imports
import numpy as np

from functions import cross_talk as ct
from functions import delta_t
from functions import fits as gf
from functions import plot_valid, spectro_stuff

# =========================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =========================================================================

path_NL11_693_12 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/Ne_693"

path_NL11_703_12 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/Ne_703"

path_NL11_703 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2208"

path_A5_SPDC_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer"

path_A5_Ar_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/Ar"

path_A5_703_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_703"

path_A5_693_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_693"

path_A5_SPDC = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/SPDC"

path_test = "C:/Users/bruce/Downloads/Ar/Ar"

# =========================================================================
# Function execution.
# =========================================================================

# delta_t.deltas_save_numpy(
#     path_NL11_693_12,
#     pix=(24, 223),
#     rewrite=True,
#     board_number="NL11",
#     timestamps=400,
#     fw_ver="2212b",
# )

# delta_t.delta_cp(
#     path_NL11_693_12,
#     pix=(24, 223),
#     rewrite=True,
#     range_left=-15e3,
#     range_right=15e3,
# )

# gf.fit_wg(path_NL11_693_12, pix_pair=(24, 223), window=4e3, step=2)

plot_valid.plot_valid_FW2212(
    path_NL11_693_12, board_number="NL11", fw_ver="2212b", timestamps=400, show_fig=True
)

# x_nm = spectro_stuff.ar_spec(
#     path_test, board_number="A5", tmrl=[810.3692, 811.5366], timestamps=200
# )

# spectro_stuff.spdc_ac_save(
#     path_A5_SPDC_spec,
#     board_number="A5",
#     pix_left=np.arange(70, 142, 1),
#     pix_right=np.arange(175, 229, 1),
#     timestamps=10,
#     rewrite=True
# )

# spectro_stuff.spdc_ac_cp(path_A5_SPDC_spec, rewrite=True, show_fig=True)

# plot_valid.plot_spdc(path_A5_SPDC_spec, board_number="A5", timestamps=10)
