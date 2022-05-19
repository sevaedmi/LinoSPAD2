""" For calculating differences between all timestamps in a single acq cycle.
The difference is calculated in a single slice of 512*number_of_cycles,
therefore only the neighboring values are taken into account.
"""
import numpy as np

# =============================================================================
# Timestamp difference for all pixels; seems to be working as it should,
# finding the right indices was tricky
# =============================================================================

# TODO: insert the correct *n -> len(Data_binary_unpacked[0])
# fill a 3D (255x255x512*n) matrix with random numbers
Data_matrix_ps = np.sort(np.random.randint(2799, size=(256, 512)))  # 2799 from
# period/2.5 ns *140 = 50/2.5*140 for 20 MHz REF clock

DeltaT = np.zeros([512, 255, 255])  # because measuring HBT peaks is to work
# with the differences between all the timestamps in the original matrix
# the dimensions are number-of-pixels - 1, therefore 255

# calculate differences through the whole sensor and the whole data file
i = 0  # for 512*n data lines
for i in range(len(Data_matrix_ps[0])):
    j = 0  # for the pixel from which the subtraction is calculated
    while j < 255:
        k = j+1  # pixels substracted from the chosen one; +1 to overcome
        # the subtraction of self
        while k < 256:  # 256 due to k-1
            # sign does not matter, hence absolute value
            DeltaT[i][j][k-1] = np.abs(Data_matrix_ps[j][i]
                                       - Data_matrix_ps[k][i])
            k = k+1  # next pixel: 0-1, 0-2,...
        j = j+1  # next row: 0-1, 1-2, 2-3,...

# =============================================================================
# Test on 2x3x3 matrix; confirmed the correctness of the 'for' loop
# =============================================================================
Data_test = np.array([[1, 21], [3, 41], [5, 71], [7, 111]])
Delta_test = np.zeros([2, 3, 3])

i = 0
for i in range(len(Data_test[0])):
    j = 0
    while j < 3:
        k = j+1
        # n = 0
        while k < 4:
            Delta_test[i][j][k-1] = np.abs(Data_test[j][i] - Data_test[k][i])
            k = k+1
        j = j+1

# TODO: differences are calculated only in a single slice of the 512*N_of_cycles
# lines of data without any check if the timestamps are close to each other.
# Need to add a filter to find close timestamps between all pixels or calculate
# differences between all timestamps in a single acq cycle, maybe in a window
# with width ~ needed time difference for HBT peaks.