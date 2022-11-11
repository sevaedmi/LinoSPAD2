import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from functions import unpack as f_up

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "Ne lamp/FW 2208/540 nm"
)

pix = (156, 157, 158, 159, 160)


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


os.chdir(path)

lines_of_data = 512

show_fig = True

DATA_FILES = glob.glob("*.dat*")

for num, filename in enumerate(DATA_FILES):
    print(
        "================================\n"
        "Working on {}\n"
        "================================".format(filename)
    )
    data = f_up.unpack_binary_flex(filename)

    data_1 = data[pix[0]]  # 1st pixel
    data_2 = data[pix[1]]  # 2nd pixel
    data_3 = data[pix[2]]  # 3d pixel
    data_4 = data[pix[3]]  # 4th pixel
    data_5 = data[pix[4]]  # 5th pixel

    pixel_numbers = np.arange(pix[0], pix[-1] + 1, 1)

    all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))

    plt.rcParams.update({"font.size": 20})
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))

    for q in range(5):
        for w in range(5):
            if w <= q:
                continue
            if np.abs(w - q) > 1:
                continue
            data_pair = np.vstack((all_data[q], all_data[w]))

            minuend = len(data_pair)
            timestamps_total = len(data_pair[0])
            subtrahend = len(data_pair)
            timestamps = lines_of_data

            output = []

            for i in tqdm(range(minuend)):
                acq = 0  # number of acq cycle
                for j in range(timestamps_total):
                    if data_pair[i][j] == -1:
                        continue
                    if j % lines_of_data == 0:
                        acq = acq + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 53-45
                        for p in range(timestamps):
                            n = lines_of_data * (acq - 1) + p
                            if data_pair[k][n] == -1:
                                continue
                            elif data_pair[i][j] - data_pair[k][n] > 3.5e3:
                                continue
                            elif data_pair[i][j] - data_pair[k][n] < -3.5e3:
                                continue
                            else:
                                output.append(data_pair[i][j] - data_pair[k][n])
            if "Ne" and "540" in path:
                chosen_color = "seagreen"
            elif "Ne" and "656" in path:
                chosen_color = "orangered"
            elif "Ar" in path:
                chosen_color = "mediumslateblue"
            else:
                chosen_color = "salmon"
            if show_fig is True:
                plt.ion()
            else:
                plt.ioff()
            try:
                bins = np.arange(np.min(output), np.max(output), 17.857)
            except Exception:
                continue
            axs[q][w - 1].set_xlabel("\u0394t [ps]")
            axs[q][w - 1].set_ylabel("Timestamps [-]")
            n, b, p = axs[q][w - 1].hist(output, bins=bins, color=chosen_color)

            par, covariance = curve_fit(gauss, b[:-1], n)
            fit_plot = gauss(b, par[0], par[1], par[2], par[3])

            plt.figure(figsize=(16, 10))
            plt.xlabel("\u0394t [ps]")
            plt.ylabel("Timestamps [-]")
            plt.plot(b[:-1], n, "o", color=chosen_color, label="data")
            plt.plot(
                b,
                fit_plot,
                "-",
                color="cadetblue",
                label="fit\n\u03BC={p1} ps\n"
                "\u03C3={p2} ps".format(
                    p1=format(par[1], ".2f"), p2=format(par[-1], ".2f")
                ),
            )
            plt.legend(loc="best")

            try:
                os.chdir("results/gauss_fit")
            except Exception:
                os.mkdir("results/gauss_fit")
                os.chdir("results/gauss_fit")
            plt.savefig("{}_fit.png".format(filename))
            plt.pause(0.1)
            os.chdir("../..")
