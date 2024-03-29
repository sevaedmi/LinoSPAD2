import os
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyarrow import feather as ft

from LinoSPAD2.functions import utils


def sigma_of_count_spread_to_average(path, pixels: list):

    os.chdir(path)

    ft_file = glob("*.feather")[1]

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    step = 10
    bins = np.arange(np.min(data_cut), np.max(data_cut), 17.857 * step)

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 17.857 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    try:
        os.chdir("results/bckg_spread")
    except Exception:
        os.makedirs("results/bckg_spread")
        os.chdir("results/bckg_spread")

    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts)
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.savefig("Background_histogram.png")

    counts_spread, bin_edges_spread = np.histogram(counts, bins=24)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=24)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)
    plt.savefig("Background_histogram_triple.png")

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(bin_centers_spread, counts_spread, label="Spread of counts")
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
    )
    ax.set_title(
        f"Ratio of spread to average: {pars[2] / np.mean(counts) * 100:.1f} %"
    )
    ax.set_xlabel("Spread [-]")
    ax.set_ylabel("Counts [-]")
    ax.text(
        0.07,
        0.9,
        f"\u03C3={pars[2]:.2f}\u00B1{np.sqrt(covs[2,2]):.2f}",
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )
    plt.savefig("Spread_fitted.png")


path = r"/media/sj/King4TB/LS2_Data/Halogen HBT/96_tmsps/delta_ts_data"
sigma_of_count_spread_to_average(path, pixels=[144, 171])
