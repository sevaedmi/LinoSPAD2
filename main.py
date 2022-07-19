""" The main hub for analyzis of data from LinoSPAD2.

Following modules can be used:

    * cross_talk - calculation of the cross-talk rate
    * cross_talk_plot - plots the cross-talk rate distribution in the
    LinoSPAD2 pixels
    * cross_talk_fast - 4-times faster script for calcultion of the cross-talk
    rate that does not work with the pixel coordinates
    * differences - calculation of the differences between all timestamps
    which can be used to calculate the Hanbury-Brown and Twiss peaks
    * td_plot - plot a histogram of timestamp differences from LinoSPAD2
    * plot_valid_timestamps - plots number of valid timestamps in each pixel

"""

from functions import cross_talk
from functions import cross_talk_plot
from functions import cross_talk_fast

from functions import differences
from functions import td_plot
from functions import delta_t_grid

from functions import plot_valid_timestamps

from functions import single_pixel_hist

# =============================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =============================================================================

path_ext_trig = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/useful data/10 lines of data/binary/Ne lamp ext trig/"\
    "setup 2/3 ms acq window"
    
path_399 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
    "3.99 ms acq window"
    
path_3 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/useful data/10 lines of data/binary/Ne lamp ext trig/setup 2/"\
    "3 ms acq window"

# =============================================================================
# Function execution.
# =============================================================================

# plot_valid_timestamps.plot_valid_per_pixel(path_3, lod=512, scale='log')

# differences.timestamp_diff_flex(path_int_clock, lod=512)

# td_plot.plot_diff(path_int_clock_res, show_fig=True)

delta_t_grid.plot_grid(path=path_399, show_fig=False)
