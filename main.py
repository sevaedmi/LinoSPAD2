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

path_NL11_703_12 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/Ne_703"

path_NL11_703 = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2208"

path_A5_SPDC_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer"

path_A5_Ar_spec = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/Ar"

path_A5_703_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_703"

path_A5_693_12 = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_693"

# =========================================================================
# Function execution.
# =========================================================================

# delta_t.deltas_save_numpy(
#     path_NL11_703_12,
#     pix=(17, 222),
#     rewrite=True,
#     board_number="NL11",
#     timestamps=140,
#     fw_ver="2212b",
# )

# delta_t.delta_cp(
#     path_NL11_703_12,
#     pix=(17, 222),
#     rewrite=True,
#     range_left=-15e3,
#     range_right=15e3,
# )

gf.fit_wg(path_NL11_703_12, pix_pair=(17, 222), window=4e3, step=3)

# plot_valid.plot_valid_FW2212_mult(
#     path_NL11_703_12, board_number="NL11", fw_ver="2212b", timestamps=140, show_fig=True
# )


# spectro_stuff.ar_spec(path_A5_Ar_spec, board_number="A5", timestamps=200)

# spectro_stuff.spdc_ac(
#     path_A5_SPDC_spec,
#     board_number="A5",
#     pix_left=[95, 96],
#     pix_right=[220, 221],
#     timestamps=20,
# )
