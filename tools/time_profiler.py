""" Time profiler: for analyzing time-consumption of codes

"""
from functions.unpack import unpack_binary_df
from functions.plot_valid import plot_valid_df
from functions.calc_diff import calc_diff
import os
import glob
import numpy as np

path = "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"\
    "BNL-Jakub/SPDC"
os.chdir(path)
filename = glob.glob("*.dat*")[0]

data = unpack_binary_df(filename, cut_empty=False)
data = unpack_binary_df(filename, cut_empty=True)

dp1 = data.Timestamp[data.Pixel == 135]
dp2 = data.Timestamp[data.Pixel == 144]

data_pair = np.vstack((dp1, dp2))

plot_df = plot_valid_df(path)

# > > > > > Using magic function timeit for quick analysis < < < < <
%timeit calc_diff(data_pair)

# > > > > > Using external module line_profiler for line-by-line analysis < < < < <
%load_ext line_profiler

%lprun -f plot_valid_df plot_valid_df(path)

# > > > > > Using line profiler library < < < < <
from line_profiler import LineProfiler

# lp = LineProfiler()
# lp_wrapper = lp(delta_t.plot_grid(path_BNL, (87, 88, 223), show_fig=True,
#                                   same_y=True))
# lp.print_stats()