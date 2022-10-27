""" Time profiler: for analyzing time-consumption of codes

"""
from functions.unpack import unpack_binary_df, unpack_binary_flex
from functions.plot_valid import plot_valid_df
from functions.calc_diff import calc_diff_df, calc_diff
import os
import glob
import numpy as np

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "BNL-Jakub/SPDC"
os.chdir(path)
filename = glob.glob("*.dat*")[0]

data = unpack_binary_flex(filename)
data_df = unpack_binary_df(filename, cut_empty=True)

#Standard
data_pair = np.vstack((data[134], data[143]))

%timeit deltas = calc_diff(data_pair)

# Dataframe
dp1 = data_df[data_df.Pixel == 135]
dp2 = data_df[data_df.Pixel == 144]

%timeit deltas_df = calc_diff_df(dp1, dp2)

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