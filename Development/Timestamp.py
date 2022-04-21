import numpy as np
import os

os.chdir("C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/2 cycles, box off")

# The .txt output of the LinoSPAD2 is a single column of data, where first 512lines are 
# the data from the 1st pixel collected during the first cycle of the data acquisition,
# next 512 lines (2*512:2*512+512) - 2nd pixel and so on. Lines 256*512:256*512+512 are 
# the data from the 1st pixel collected during the 2nd cycle.
Data = np.genfromtxt("acq_220420_122633.txt")

Data_matrix = np.zeros((256, int(len(Data)/256))) # rows=#pixels, cols=#cycles

noc = len(Data)/512/256 # number of cycles, 512 data lines per pixel per cycle, 256 pixels

# =============================================================================
# Unpack data from the txt, which is a Nx1 array, into a 256x#cycles matrix using two 
# counters. When i (number of the pixel) reaches the 255 (256 pixel), move k one step
# further - second data acquisition cycle, when the algorithm goes back to the 1st pixel
# and writes the data right next to the data from previous cycle.
# =============================================================================
k=0
while k!=noc:
    i=0
    while i<256:
        Data_matrix[i][k*512:k*512+512] = Data[(i+256*k)*512:(i+256*k)*512+512]-2**31
        i=i+1
    k=k+1
