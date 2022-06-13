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

"""

# from functions import cross_talk
# from functions import cross_talk_plot
# from functions import cross_talk_fast
# from functions import differences
from functions import td_plot

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

# =============================================================================
# Function execution.
# =============================================================================

# Calculate cross-talk rate in %, save the result in a .csv file
# cross_talk.cross_talk_rate(path_test)

# Plot the cross-talk rate distribution in the sensor
# cross_talk_plot.ctr_dist(path_test, show_fig=True)

# Fast calculation of cross-talk
# cross_talk_rate = cross_talk_fast.cross_talk_rate(path_test)

# Calculate timestamp differences between all pixels for the HBT peaks
# differences.timestamp_diff(path_test)

# Plot a histogram of timestamp differences
td_plot.plot_diff(path_res_test, show_fig=True)
