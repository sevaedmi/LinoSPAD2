import os
import glob
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from functions import unpack as f_up
from functions.calc_diff import calculate_differences as cd

path = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software"
    "/Data/BNL-Jakub/SPDC"
)

show_fig = True
same_y = True

timestamps = 512
pix = np.array((87, 223))

if show_fig is True:
    plt.ion()
else:
    plt.ioff()
os.chdir(path)

DATA_FILES = glob.glob("*.dat*")

for num, filename in enumerate(DATA_FILES):

    print(
        "=====================================================\n"
        "Plotting a delta t grid, Working on {}\n"
        "=====================================================\n".format(filename)
    )

    data = f_up.unpack_binary_flex(filename, timestamps)

    data_pix = np.zeros((len(pix), len(data[0])))

    for i, num1 in enumerate(pix):
        data_pix[i] = data[num1]
    plt.rcParams.update({"font.size": 22})
    ax1, ax2 = len(pix) - 1, len(pix) - 1
    # fig, axs = plt.subplots(len(pix)-1, len(pix)-1, figsize=(24, 24)
    fig, axs = plt.subplots(ax1, ax2)

    # check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0
    print("\n> > > Calculating the timestamp differences < < <\n")
    for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
        for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
            if w <= q:
                continue
            data_pair = np.vstack((data_pix[q], data_pix[w]))

            delta_ts = cd(data_pair, timestamps=timestamps)

            if "Ne" and "540" in path:
                chosen_color = "seagreen"
            elif "Ne" and "656" in path:
                chosen_color = "orangered"
            elif "Ar" in path:
                chosen_color = "mediumslateblue"
            else:
                chosen_color = "salmon"
            try:
                bins = np.arange(np.min(delta_ts), np.max(delta_ts), 17.857 * 2)
            except Exception:
                print("Couldn't calculate bins: probably not enough delta ts.")
                continue
            axs[q][w - 1].set_xlabel("\u0394t [ps]")
            axs[q][w - 1].set_ylabel("Timestamps [-]")
            n, b, p = axs[q][w - 1].hist(delta_ts, bins=bins, color=chosen_color)[0]
            # find position of the histogram peak
            try:
                n_max = np.argmax(n)
                arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
            except Exception:
                arg_max = None
            if same_y is True:
                try:
                    y_max = np.max(n)
                except ValueError:
                    y_max = 0
                    print("\nCould not find maximum y value\n")
                if y_max_all < y_max:
                    y_max_all = y_max
                axs[q][w - 1].set_ylim(0, y_max + 4)
            axs[q][w - 1].set_xlim(-2.5e3, 2.5e3)

            axs[q][w - 1].set_title(
                "Pixels {p1}-{p2}\nPeak position {pp}".format(
                    p1=pix[q], p2=pix[w], pp=arg_max
                )
            )
    if same_y is True:
        for q in range(len(pix)):
            for w in range(len(pix)):
                if w <= q:
                    continue
                axs[q][w - 1].set_ylim(0, y_max_all + 10)
    try:
        os.chdir("results/delta_t")
    except FileNotFoundError:
        os.mkdir("results/delta_t")
        os.chdir("results/delta_t")
    fig.tight_layout()  # for perfect spacing between the plots
    plt.savefig("{name}_delta_t_grid.png".format(name=filename))
    os.chdir("../..")
