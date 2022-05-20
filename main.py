""" The main hub for analyzis of data from LinoSPAD2.

"""

from functions.cross_talk import cross_talk
# insert the path to where the data are located
path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/"\
    "Data/40 ns window, 20 MHz clock, 10 cycles/10 lines of data/txt"

cross_talk.cross_talk_rate(path)
