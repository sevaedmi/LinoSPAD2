""" A script for running a memory profiler with saving the output to .txt.
This approach could be used for single lines of codes as well as for a whole
function.

"""

from functions import fits, delta_t
import glob
import os

path_v_585 = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585"

# load the memory profiler for the magic function mprun
%load_ext memory_profiler

# '-T' for saving output, in this case it is saved to a "profiler" folder in the
# path_v_585. '-f' for function which should be analyzed, followed by the mentioned function
# with all the parameters

%mprun -T profiler/plot_grid_calib_mult -f delta_t.plot_grid_calib_mult delta_t.plot_grid_calib_mult(path_v_585,pix=(225, 226, 236, 237),board_number="A5",range_left=-15e3,range_right=15e3,timestamps=80,mult_files=True)

# decode the output to a readable .txt
os.chdir("profiler")
file = glob.glob("*plot_grid_calib_mult*")[0]
with open(file, "rb") as bin_file, open("{}.txt".format(file), "w") as text_file:
    text_file.write(bin_file.read().decode())
