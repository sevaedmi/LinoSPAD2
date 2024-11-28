from pyarrow import feather as ft
import os
from matplotlib import pyplot as plt

file1 = r"2024-11-27_14-15-00_merged.feather"
file2 = r"2024-11-27_14-19-43_merged.feather"

data1 = ft.read_feather(file1).dropna()
data2 = ft.read_feather(file2).dropna()

plt.figure()
plt.hist(data1, bins=100)
plt.figure()
plt.hist(data2, bins=100)
plt.show()
