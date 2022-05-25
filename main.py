""" The main hub for analyzis of data from LinoSPAD2.

Following modules can be used:

    * cross_talk - calculation of the cross-talk rate
    * differences - calculation of the differences between all timestamps
                    which can be used to calculate the Hanbury-Brown and
                    Twiss peaks

"""

from functions import cross_talk
# from functions.differences import differences

# insert the path to where the data are located
path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/binary"

# Calculate cross-talk rate in %, save the result in a .csv file
cross_talk.cross_talk_rate(path)

# Calculate timestamp differences between all pixels for the HBT peaks
# differences.timestamp_diff(path)
