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

path_NL11_noise_check = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/noise_check"

path_NL11_703_12 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/Ne_703"

path_A5_SPDC_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer"

path_A5_Ar_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/Ar"

# =========================================================================
# Function execution.
# =========================================================================

# mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

# delta_t.deltas_save(
#     path_NL11_703_12,
#     pix=(8, 9, 10, 14, 15, 16, 26, 27, 28, 29, 30),
#     rewrite=True,
#     board_number="NL11",
#     timestamps=1000,
#     fw_ver="2212b",
# )

# delta_t.delta_cp(
#     path_A5_SPDC_spec,
#     pix=(30, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140),
#     rewrite=True,
#     range_left=-10e3,
#     range_right=10e3,
# )

# # gf.fit_wg(path_693_12, pix_pair=(3, 45), window=4e3)

# plot_valid.plot_valid_FW2212_mult(
#     path_A5_Ar_spec,
#     board_number="A5",
#     fw_ver="block",
#     timestamps=160,
#     show_fig=False,
#     app_mask=True
# )

plot_valid.plot_pixel_hist(
    path_NL11_703_12,
    pix1=[150, 215],
    fw_ver="2212b",
    board_number="NL11",
    timestamps=1000,
)
