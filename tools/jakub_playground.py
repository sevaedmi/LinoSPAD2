import numpy as np
from functions import delta_t, plot_valid, fits as gf
from functions._old import delta_t_single_plots
import functions.delta_t
import matplotlib as plt
from matplotlib import pyplot as plt
from functions import unpack as f_up


def main_playground():
    # First peak 84 85 86 87 88 89
    # Second peak 218 219 220 221 222
    ## hist two pixels
    # path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221017/no_delay/Analyze/acq_221017_155542.dat"
    # path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221017/no_delay/acq_221017_155636.dat"
    path_to_file = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221017/added_2m_optical_cable_in_IDLER/After_focused/acq_221018_000228.dat"
    time_window = 25000

    data_ps_float = np.round(f_up.unpack_binary_flex(path_to_file, 512))
    data_ps_int = data_ps_float.astype(np.int64)

    pixel_84 = data_ps_int[84]
    pixel_85 = data_ps_int[85]
    pixel_86 = data_ps_int[86]
    pixel_87 = data_ps_int[87]

    pixel_218 = data_ps_int[218]
    pixel_219 = data_ps_int[219]
    pixel_220 = data_ps_int[220]
    pixel_221 = data_ps_int[221]
    pixel_222 = data_ps_int[222]
    pixel_223 = data_ps_int[223]


    # delta_t_85_218 = functions.delta_t.compute_delta_t(pixel_85, pixel_218, 512, timewindow=time_window)
    # print("A")
    # delta_t_85_219 = functions.delta_t.compute_delta_t(pixel_85, pixel_219, 512, timewindow=time_window)
    # delta_t_85_220 = functions.delta_t.compute_delta_t(pixel_85, pixel_220, 512, timewindow=time_window)
    # delta_t_85_221 = functions.delta_t.compute_delta_t(pixel_85, pixel_221, 512, timewindow=time_window)
    # delta_t_85_222 = functions.delta_t.compute_delta_t(pixel_85, pixel_222, 512, timewindow=time_window)
    # print("B")
    # delta_t_86_218 = functions.delta_t.compute_delta_t(pixel_86, pixel_218, 512, timewindow=time_window)
    # delta_t_86_219 = functions.delta_t.compute_delta_t(pixel_86, pixel_219, 512, timewindow=time_window)
    # delta_t_86_220 = functions.delta_t.compute_delta_t(pixel_86, pixel_220, 512, timewindow=time_window)
    # delta_t_86_221 = functions.delta_t.compute_delta_t(pixel_86, pixel_221, 512, timewindow=time_window)
    # delta_t_86_222 = functions.delta_t.compute_delta_t(pixel_86, pixel_222, 512, timewindow=time_window)
    # print("C")
    # delta_t_87_218 = functions.delta_t.compute_delta_t(pixel_87, pixel_218, 512, timewindow=time_window)
    # delta_t_87_219 = functions.delta_t.compute_delta_t(pixel_87, pixel_219, 512, timewindow=time_window)
    # delta_t_87_220 = functions.delta_t.compute_delta_t(pixel_87, pixel_220, 512, timewindow=time_window)
    # delta_t_87_221 = functions.delta_t.compute_delta_t(pixel_87, pixel_221, 512, timewindow=time_window)
    # delta_t_87_222 = functions.delta_t.compute_delta_t(pixel_87, pixel_222, 512, timewindow=time_window)
    # print("D")
    # delta_t_88_218 = functions.delta_t.compute_delta_t(pixel_88, pixel_218, 512, timewindow=time_window)
    # delta_t_88_219 = functions.delta_t.compute_delta_t(pixel_88, pixel_219, 512, timewindow=time_window)
    # delta_t_88_220 = functions.delta_t.compute_delta_t(pixel_88, pixel_220, 512, timewindow=time_window)
    # delta_t_88_221 = functions.delta_t.compute_delta_t(pixel_88, pixel_221, 512, timewindow=time_window)
    # delta_t_88_222 = functions.delta_t.compute_delta_t(pixel_88, pixel_222, 512, timewindow=time_window)
    #
    # delta_t = np.hstack((delta_t_84_218, delta_t_84_219, delta_t_84_220, delta_t_84_221, delta_t_84_222,
    #                      delta_t_85_218, delta_t_85_219, delta_t_85_220, delta_t_85_221, delta_t_85_222,
    #                      delta_t_86_218, delta_t_86_219, delta_t_86_220, delta_t_86_221, delta_t_86_222,
    #                      delta_t_87_218, delta_t_87_219, delta_t_87_220, delta_t_87_221, delta_t_87_222))
    # Creating histogram
    delta_t_87_223 = functions.delta_t.compute_delta_t(pixel_87, pixel_223, 512, timewindow=time_window)

    bins = np.arange(np.min(delta_t_87_223), np.max(delta_t_87_223),
                     17.857 * 2)

    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)
    axs.hist(delta_t_87_223, bins=bins)
    plt.show()
    #
    # # # plot delta T plot
    data_path = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221017/added_2m_optical_cable_in_IDLER/After_focused/Analyze_big"
    functions.delta_t.plot_grid(data_path, (86,87,88, 222,223,224), 512, show_fig=False, same_y=True)

    # # Plot histogram of incoming timestamps per pixel with
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
