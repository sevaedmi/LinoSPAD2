""" Time profiler: for analyzing time-consumption of codes

"""
from functions.unpack import unpack_binary_df, unpack_binary_flex
from functions.plot_valid import plot_valid_df, plot_valid
from functions.delta_t import plot_grid_df, plot_grid
from functions.calc_diff import calc_diff_df, calc_diff
import os
import glob
import numpy as np

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/Ne lamp/FW 2208/540 nm"
os.chdir(path)
filename = glob.glob("*.dat*")[0]

# Unpacking

data = unpack_binary_flex(filename)
data_df = unpack_binary_df(filename, cut_empty=True)

# Standard plot_valid

%timeit plot_valid(path, pix=(149, 150, 151), mask=[], timestamps=512)

# Dataframe plot_valid_df

%timeit plot_valid_df(path)

# Standard calc_diff
data_pair = np.vstack((data[134], data[143]))

%timeit deltas = calc_diff(data_pair)

# Dataframe calc_diff_df
dp1 = data_df[data_df.Pixel == 135]
dp2 = data_df[data_df.Pixel == 144]

%timeit deltas_df = calc_diff_df(dp1, dp2)

# Standard plot_grid
pix = (155, 156, 157, 158, 159, 160)
%timeit plot_grid(path, pix)

# Dataframes plot_grid_df
pix = (155, 156, 157, 158, 159, 160)
%timeit plot_grid_df(path, pix)

# > > > > > Using magic function timeit for quick analysis < < < < <
%timeit calc_diff(data_pair)

# > > > > > Using external module line_profiler for line-by-line analysis < < < < <
%load_ext line_profiler

%lprun -T profiler/calc_diff_df1 -f calc_diff_df calc_diff_df(dp1, dp2)

os.chdir("profiler")
file = "calc_diff_df1"
with open("{}".format(file), "rb") as bin_file, open("{}.txt".format(file), "w") as text_file:
    text_file.write(bin_file.read().decode())
os.chdir("..")

# > > > > > Using line profiler library < < < < <
# from line_profiler import LineProfiler

# lp = LineProfiler()
# lp_wrapper = lp(delta_t.plot_grid(path_BNL, (87, 88, 223), show_fig=True,
#                                   same_y=True))
# lp.print_stats()