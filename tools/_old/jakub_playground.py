import numpy as np
from scipy.stats import norm
from functions import delta_t, plot_valid, fits as gf
from functions._old import delta_t_single_plots
import functions.delta_t
import matplotlib as plt
from matplotlib import pyplot as plt
from functions import unpack as f_up
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from struct import unpack
import time
# import torch
import socket

# For the remote control you have to send single-line commands. Each line
# should end with '\n' and the program responds with 'DONE' when the
# command executed. Commands are:
# RUNPREVIEW
# SAVEPREVIEW <filename>
# RUNCOUNTERS
# SAVECOUNTERS <filename>
# RUNHISTOGRAMS
# SAVEHISTOGRAMS <filename>
# LOADCONFIG <filename>
#
# The '...HISTOGRAMS' commands correspond to the buttons in the 'TDC data
# acquisition' tab.


def tcp_test():
    print("TCP")
    TCP_IP = '127.0.0.1'
    TCP_PORT = 1029
    BUFFER_SIZE = 1024
    MESSAGE = "SAVEHISTOGRAMS testsvahistrohramda.dat\n"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.send(MESSAGE.encode())
    data = s.recv(BUFFER_SIZE)
    s.close()
    print("received data:", str(data))


def main_playground():
    tcp_test()


    # filename = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221021/acq_221021_202311_timeStamps512.dat"
    # # filename = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221028_calibration/acq_221028_171552.dat"  # DONE OK
    # filename = "C:/Users/jakub/Documents/LinoSpad2/data/SPDC_221021/pixel_movement/acq_221021_212530_199_206.dat"  # DONE OK
    # lines_of_data = 512
    # timestamp_list = []
    # address_list = []
    #
    # st = time.time()
    # f = open(filename, "r")
    # a = np.fromfile(f, dtype=np.uint32)  # uint32
    # data = (a & 0xFFFFFFF).astype(int) * 17.857
    # data[np.where(a < 0x80000000)] = -1
    # data = data[0 : lines_of_data * 3 * 256]
    # print(a[0])
    # noc = int(len(data) / lines_of_data / 256)  # number of cycles,
    #
    # # b = data.reshape((lines_of_data,256*noc)).transpose()#.reshape((lines_of_data*noc,256),order='F').transpose()
    # #
    # # B = np.reshape(np.transpose(np.reshape(b, (512,256,-1)), (0, 2, 1)),(-1,256))
    # #
    # t0 = data.reshape((lines_of_data, 256 * noc), order="F")
    #
    # t1 = np.reshape(data, (lines_of_data, noc * 256), order="F")
    # t2 = np.reshape(t1, (lines_of_data, 256, -1), order="F")
    # t3 = np.transpose(t2, (0, 2, 1))
    # t4 = np.reshape(t2.transpose((0, 2, 1)), (-1, 256), order="F")
    #
    # # t3 = np.transpose(t1, (0, 2, 1))
    # # t4= np.reshape(t2, (-1, 256))
    #
    # print(t4.transpose()[0:2, 0:3])
    # t2.transpose(())
    #
    # et = time.time()
    # print("Execution time:", et - st, "seconds")
    #
    # st = time.time()
    # with open(filename, "rb") as f:
    #     while True:
    #         rawpacket = f.read(4)  # read 32 bits
    #         if not rawpacket:
    #             break  # stop when the are no further 4 bytes to readout
    #         packet = unpack("<I", rawpacket)
    #         if (packet[0] >> 31) == 1:  # check validity bit: if 1
    #             # - timestamp is valid
    #             timestamp = packet[0] & 0xFFFFFFF  # cut the higher bits,
    #             # timestamp = packet[0] & 0xFF  # cut the higher bits,
    #             # leave only timestamp ones
    #             # 2.5 ns from TDC 400 MHz clock read out 140 bins from 35
    #             # elements of the delay line - average bin size is 17.857 ps
    #             timestamp = timestamp * 17.857  # in ps
    #         else:
    #             timestamp = -1
    #         timestamp_list.append(timestamp)
    #         address = (packet[0] >> 28) & 0x3  # gives away only zeroes -
    #         # not in this firmware??
    #         address_list.append(address)
    # # rows=#pixels, cols=#cycles
    # data_matrix = np.zeros((256, int(len(timestamp_list) / 256)))
    #
    # noc = len(timestamp_list) / lines_of_data / 256  # number of cycles,
    # # lines_of_data data lines per pixel per cycle, 256 pixels
    #
    # # pack the data from a 1D array into a 2D matrix
    # k = 0
    # while k != noc:
    #     i = 0
    #     while i < 256:
    #         data_matrix[i][
    #             k * lines_of_data : k * lines_of_data + lines_of_data
    #         ] = timestamp_list[
    #             (i + 256 * k) * lines_of_data : (i + 256 * k) * lines_of_data
    #             + lines_of_data
    #         ]
    #         i = i + 1
    #     k = k + 1
    # et = time.time()
    # print("Execution time:", et - st, "seconds")
    # print("Execution time:", et - st, "seconds")

    # data = np.random.normal(loc=5.0, scale=2.0, size=1000)
    # mean, std = norm.fit(data)
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
