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
from functions import plot_valid_timestamps

# =============================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =============================================================================

# main path where the binary data collected with LinoSPAD2 are
path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary"

# path to small data files for codes testing
path_test = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "for codes"

# path to the interstep results from small data files for code testing
path_res_test = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data"\
    "/binary/for codes/results"

# path to ref data with sensor cover, w/o the fiber on the sensor and 100
# lines of data

path_ref = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary"\
    "/w cover wo fiber"

path_ref_res = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"\
    "/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary"\
    "/w cover wo fiber/results"

# path to the data files from Ne lamp w 540 nm filter, 100 data lines per acq
# window

path_Ne_w_100 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/lamp w filter/100 data lines"

# path to data from setup with an external trigger of 250 Hz

path_ext_trig = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/Ext trig test"

plot_test_30 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary/"\
    "Ne lamp ext trig/30 lines of data"

plot_test_512 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/Ne lamp ext trig"

plot_test_30_2 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/Ne lamp ext trig/setup 2/30 lines of data"

plot_test_512_2 = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/"\
    "Software/Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/"\
    "binary/Ne lamp ext trig/setup 2"

# =============================================================================
# Function execution.
# =============================================================================

# Calculate cross-talk rate in %, save the result in a .csv file
# cross_talk.cross_talk_rate(path_test)

# Plot the cross-talk rate distribution in the sensor
# cross_talk_plot.ctr_dist(path_test, show_fig=True)

# Fast calculation of cross-talk
# cross_talk_rate = cross_talk_fast.cross_talk_rate(path_ext_trig)

# Calculate timestamp differences between all pixels for the HBT peaks
differences.timestamp_diff_flex(plot_test_512, 512)

# Plot a histogram of timestamp differences
# td_plot.plot_diff(path_ext_trig, show_fig=True)

# Plot number of valid timestamps in each pixel
# plot_valid_timestamps.plot_valid_per_pixel(plot_test_512, lod=512)
# plot_valid_timestamps.plot_valid_per_pixel(plot_test_512, lod=512)
# plot_valid_timestamps.plot_valid_per_pixel(plot_test_30_2, lod=30)
# plot_valid_timestamps.plot_valid_per_pixel(plot_test_512_2, lod=512)
