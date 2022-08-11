""" The main hub for analyzis of data from LinoSPAD2.

Following modules can be used:

    * cross_talk_flex - calculation of the average cross-talk rate for
    neighboring pixels
    * cross_talk_flex_over_1 - calculation of the average cross-talk rate for
    pixels with one in between (i, i+2)
    * delta_t_grid - plots a grid of 4x4 of timestamps differences between 5
    pixels in the given range
    * plot_valid_timestamps - plots number of valid timestamps in each pixel
    * single_pixel_hist - plots a separate histograms of timestamps for pixels
    in the giver ranges

"""
import numpy as np

from functions import cross_talk_flex
from functions import cross_talk_flex_over_1

from functions import delta_t_grid

from functions import plot_valid_timestamps

from functions import single_pixel_hist

from functions import online_plot_valid

# =============================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =============================================================================

path_3_99 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/Ne lamp ext trig/setup 2/3.99 ms acq window"

# path_3_99_ = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
#     "Data/Ne lamp ext trig/setup 2/3.99 ms acq window/656 nm"

path_ct = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/useful data/10 lines of data/binary"

path__3_99 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/Ne lamp ext trig/setup 2/FW 2208/3.99 ms"

path_test = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/useful data/10 lines of data/binary"

path_FW_3_99 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/Ne lamp ext trig/setup 2/FW 2208/3.99 ms"

# =============================================================================
# Function execution.
# =============================================================================

# delta_t_grid.plot_grid(path_FW_3_99, (155, 156, 157, 158, 159), True)

# plot_valid_timestamps.plot_valid_per_pixel(path_FW_3_99, 512, 'log', True)

# cross_talk_flex.cross_talk_rate(path_3_99, 512)

# ct_rate_over_1 = cross_talk_flex_over_1.cross_talk_rate(path_3_99, 512)

pix_range = np.arange(145, 166, 1)
online_plot_valid.online_plot_valid(path_FW_3_99, pix_range)
