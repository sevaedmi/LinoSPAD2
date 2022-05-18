# =============================================================================
# Test script for checking LinoSPAD2 pixel crosstalk
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt

# Test with random numbers
Data_matrix = np.sort(np.random.randint(559, size=(256, 512)))
Diff_matrix = np.zeros((255, 512))

# calculate time differences between two neighboring rows
for i in range(len(Data_matrix)-1):
    Diff_matrix[i] = Data_matrix[i+1] - Data_matrix[i]

Diff_counts_matrix = np.zeros((255, 21))
# count the differences
for i in range(len(Diff_matrix)):
    for j in range(-10, 11):
        Diff_counts_matrix[i][j+10] = len(np.where(Diff_matrix[i] == j)[0])

# =============================================================================

# Plot to check the results
plt.figure(figsize=(16, 10))
plt.xlabel("Time difference [ps]")
plt.ylabel("Counts [-]")
for i in range(len(Diff_counts_matrix)):
    # for j in range (0, 20):
    #         plt.plot(j-10, Diff_counts_matrix[i][j], 'o')
    plt.plot(np.arange(-10, 11, 1), Diff_counts_matrix[i])
plt.show()

# Calculate crosstalk in %; at [10] lie time differences of 0
CT = np.zeros(255)
for i in range(len(CT)):
    CT[i] = Diff_counts_matrix[i][10]/np.sum(Diff_counts_matrix[i])*100
