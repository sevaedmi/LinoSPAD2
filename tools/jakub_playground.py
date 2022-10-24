import numpy as np
from scipy.stats import norm
from functions import delta_t, plot_valid, fits as gf
from functions._old import delta_t_single_plots
import functions.delta_t
import matplotlib as plt
from matplotlib import pyplot as plt
from functions import unpack as f_up
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def main_playground():
    data = np.random.normal(loc=5.0, scale=2.0, size=1000)
    mean, std = norm.fit(data)
    # First peak 84 85 86 87 88 89
    # Second peak 218 219 220 221 222
    ## hist two pixels
    # path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221018/no_delay/acq_221018_230939.dat"
    # path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221018/added_2m_optical_cable_in_SIGNAL/acq_221018_231231.dat"
    # path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221018/added_2m_optical_cable_in_IDLER/acq_221018_231455.dat"
    # # path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221017/no_delay/acq_221017_155542.dat"
    #
    # time_window = 40000
    #
    # data_ps_float = np.round(f_up.unpack_binary_flex(path_to_file, 512))
    # data_ps_int = data_ps_float.astype(np.int64)
    #
    # pixel_133 = data_ps_int[133]
    # pixel_134 = data_ps_int[134]
    # pixel_135 = data_ps_int[135]
    #
    # pixel_142 = data_ps_int[142]
    # pixel_143 = data_ps_int[143]
    # pixel_144 = data_ps_int[144]
    #
    # # Creating histogram
    # delta_t_87_223 = functions.delta_t.compute_delta_t(pixel_134, pixel_143, 512, timewindow=time_window)
    #
    # # bins = np.arange(np.min(delta_t_87_223), np.max(delta_t_87_223),
    # #                   17.857 * 2)
    #
    #
    # plt.rcParams.update({"font.size": 18})
    # fig, axs = plt.subplots(1, 1,
    #                         figsize=(10, 7),
    #                         tight_layout=True)
    # axs.hist(delta_t, bins=bins)
    # axs.set_title("Coincidence rate pixels 134-143: 2 m fiber added to IDLER")
    # plt.rcParams.update({"font.size": 18})
    # plt.xlabel("Time difference [ps]")
    # plt.ylabel("Number of occurrence [-]")
    #
    # plt.box(bool(1))
    # plt.grid(False)
    # plt.subplots_adjust(left=0.15)
    #
    # axs.tick_params(which="both", width=2, direction="in")
    # axs.tick_params(which="major", length=7, direction="in")
    # axs.tick_params(which="minor", length=4, direction="in")
    # axs.yaxis.set_ticks_position("both")
    # axs.xaxis.set_ticks_position("both")
    # axs.xaxis.set_minor_locator(AutoMinorLocator())
    # axs.yaxis.set_minor_locator(AutoMinorLocator())
    # axs.set_xlim(-time_window, time_window)
    # for axis in ["top", "bottom", "left", "right"]:
    #     axs.spines[axis].set_linewidth(2)
    # plt.draw()
    # plt.show()

    # # # plot delta T plot
    # data_path = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221017/added_2m_optical_cable_in_IDLER/After_focused/Analyze_big"
    # functions.delta_t.plot_grid(data_path, (86,87,88, 222,223,224), 512, show_fig=False, same_y=True)

    # # Plot histogram of incoming timestamps per pixel
    # data_ps_float = np.round(f_up.unpack_binary_flex(path_to_file, 512))
    # data_ps_int = data_ps_float.astype(np.int64)
    # print(np.amax(data_ps_int))
    # pixel_0 = data_ps_int[30]
    # fig, axs = plt.subplots(1, 1,
    #                         figsize=(10, 7),
    #                         tight_layout=True)
    # bins = np.arange(0, 4e9, 17.867 * 1e6)
    #
    # axs.hist(pixel_0, bins=bins)
    # plt.show()





