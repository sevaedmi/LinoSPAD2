import os
import glob
import numpy as np
from functions.unpack import unpack_binary_flex
from functions.unpack import unpack_binary_flex1
from functions.plot_valid import plot_valid
import seaborn as sns

path_BNL = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"
    "/Data/BNL-Jakub/SPDC"
)

os.chdir(path_BNL)

filename = glob.glob("*.dat*")[0]

# %timeit data = unpack_binary_flex(filename, 512)
data = unpack_binary_flex(filename, 512)
# %timeit data1 = unpack_binary_flex1(filename, 512)

mask = [
    15,
    16,
    29,
    39,
    40,
    50,
    52,
    66,
    73,
    93,
    95,
    96,
    98,
    101,
    109,
    122,
    127,
    196,
    210,
    231,
    236,
    238,
]
data1 = unpack_binary_flex1(filename, 512)
data2 = data1.sort_values('Pixel')

data3 = data2[data2.Timestamp > 0]

data4 = data3[~data3['Pixel'].isin(mask)]

valid = sns.histplot(x='Pixel', data=data4)
valid.set_yscale("log")
# sns.relplot(x='Pixel', y='Timestamp', data=data3)
sns.relplot(x='Pixel', y='Timestamp', data=data3)

plot_valid(path_BNL, pix=[87,88,222,223,224], mask=[], timestamps=512, show_fig=True)

%timeit unpack_binary_flex(filename, 512)
%timeit unpack_binary_flex1(filename, 512)

%timeit plot_valid(path_BNL, pix=[87,88,222,223,224], mask=[], timestamps=512, show_fig=True)
%timeit sns.countplot(data1.Pixel)
