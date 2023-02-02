from functions import fits, delta_t
import glob
import os

path_v_585 = "D:/LinoSPAD2/Data/board_A5/V_setup/Ne_585"

%load_ext memory_profiler

%mprun -T profiler/plot_grid_calib_mult -f delta_t.plot_grid_calib_mult delta_t.plot_grid_calib_mult(path_v_585,pix=(225, 226, 236, 237),board_number="A5",range_left=-15e3,range_right=15e3,timestamps=80,mult_files=True)
# %mprun -T profiler/plot_grid_calib_mult_cut -f delta_t.plot_grid_calib_mult_cut delta_t.plot_grid_calib_mult_cut(path_v_585,pix=(225, 226, 236, 237),board_number="A5",range_left=-15e3,range_right=15e3,timestamps=80,mult_files=True)

os.chdir("profiler")
file = glob.glob("*plot_grid_calib_mult*")[0]
with open(file, "rb") as bin_file, open(
    "{}.txt".format(file), "w"
) as text_file:
    text_file.write(bin_file.read().decode())
