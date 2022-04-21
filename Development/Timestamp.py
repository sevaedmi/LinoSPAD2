import numpy as np
import os

os.chdir("C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/2 cycles, box off")

Data = np.genfromtxt("acq_220420_122633.txt")

Data_matrix = np.zeros((256, int(len(Data)/256))) # rows=#pixels, cols=#cycles

noc = len(Data)/512/256 # number of cycles, 512 data lines per pixel per cycle,\
    #256 pixels


k=0

while k!=noc:
    i=0
    while i<256:
        Data_matrix[i][k*512:k*512+512] = Data[(i+256*k)*512:(i+256*k)*512+512]-2**31
        i=i+1
    k=k+1
